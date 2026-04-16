import torch
import torch.nn.functional as F
import numpy as np
import cv2
import io
from PIL import Image
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple


@dataclass
class DiseaseHotspot:
    """A single disease region identified on the Grad-CAM heatmap."""
    xPct: float          # 0–100, left-to-right percentage on original image
    yPct: float          # 0–100, top-to-bottom percentage on original image
    intensity: float     # 0–1, how strong the activation is (1.0 = red in JET)
    radius: float        # approximate radius as % of image width
    rank: int            # 1 = most critical, 2 = secondary, etc.
    label: str           # human-readable position ("centre-leaf", "leaf-margin", etc.)
    area_pct: float      # percentage of leaf area this hotspot covers


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handlers = []

        def save_activations(module, input, output):
            self.activations = output

        def save_gradients(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.handlers.append(target_layer.register_forward_hook(save_activations))
        self.handlers.append(target_layer.register_full_backward_hook(save_gradients))

    def generate_heatmap(self, input_tensor, class_idx=None):
        """
        Generates a 2D heatmap using Grad-CAM++ logic.
        Grad-CAM++ considers higher-order gradients for better localization.
        """
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        one_hot = torch.zeros((1, output.size()[-1]), dtype=torch.float32).to(input_tensor.device)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Grad-CAM++ Implementation logic
        grads = self.gradients.detach()
        activations = self.activations.detach()

        # Compute alpha values (Grad-CAM++ optimization)
        grads_power_2 = grads**2
        grads_power_3 = grads**3
        sum_grads = torch.sum(grads, dim=(2, 3), keepdim=True)

        alpha_num = grads_power_2
        alpha_denom = grads_power_2 * 2 + sum_grads * grads_power_3
        # Avoid division by zero
        alpha_denom = torch.where(alpha_denom != 0, alpha_denom, torch.ones_like(alpha_denom))
        alphas = alpha_num / alpha_denom

        # Compute weights based on alphas and ReLU'd gradients
        weights = torch.sum(alphas * F.relu(grads), dim=(2, 3), keepdim=True)
        
        # Weighted combination of activations
        grad_cam_pp = torch.sum(weights * activations, dim=1, keepdim=True)
        grad_cam_pp = F.relu(grad_cam_pp)

        # Normalize heatmap to [0, 1]
        heatmap_min = grad_cam_pp.min()
        heatmap_max = grad_cam_pp.max()
        grad_cam_pp = (grad_cam_pp - heatmap_min) / (heatmap_max - heatmap_min + 1e-8)

        heatmap = grad_cam_pp.cpu().numpy().squeeze()
        return heatmap, class_idx

    # ─────────────────────────────────────────────────────────────────────────
    # IMPROVED: Disease-focused hotspot detection with Non-Maximum Suppression
    # ─────────────────────────────────────────────────────────────────────────

    def find_hotspots(
        self,
        heatmap: np.ndarray,
        threshold: float = 0.50,
        max_spots: int = 5,
        image_size: Optional[tuple] = None,
        nms_overlap_threshold: float = 0.35,
    ) -> List[dict]:
        """
        Identifies disease regions in the Grad-CAM heatmap with NMS deduplication.

        Improvements over v1:
          - CLAHE contrast enhancement on the heatmap before thresholding
          - Lower default threshold (0.50) to catch orange/early-lesion zones
          - Non-Maximum Suppression to avoid overlapping duplicate pins
          - Area-weighted scoring (larger lesion zones ranked higher)
          - Peak pixel position used as anchor when contour is ambiguous
          - Returns `area_pct` field for frontend badge display

        Args:
            heatmap:              2D numpy array, values in [0, 1].
            threshold:            Activation fraction for binary mask.
            max_spots:            Maximum hotspots to return.
            image_size:           (width, height) of display image.
            nms_overlap_threshold: IoU above which two circles are merged.
        """
        h, w = heatmap.shape

        # ── 1. CLAHE enhancement — brings out subtle activation zones ──
        heatmap_8u = np.uint8(heatmap * 255)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
        enhanced = clahe.apply(heatmap_8u).astype(np.float32) / 255.0

        # Blend enhanced with original to keep global structure
        blended = 0.6 * heatmap + 0.4 * enhanced
        blended = np.clip(blended, 0, 1)

        # ── 2. Smooth ──
        smoothed = cv2.GaussianBlur(blended, (11, 11), 0)
        smoothed = np.clip(smoothed, 0, 1)

        # ── 3. Binary mask at threshold ──
        binary = (smoothed >= threshold).astype(np.uint8) * 255

        # ── 4. Morphological closing — merge nearby blobs ──
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        # Remove tiny isolated specks
        open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_k)

        # ── 5. Find contours ──
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            gy, gx = np.unravel_index(np.argmax(smoothed), smoothed.shape)
            return [_make_hotspot(gx / w, gy / h, float(smoothed[gy, gx]), 0.0, 1, (h, w), image_size)]

        # ── 6. Score contours ──
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 6:
                continue

            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
            mean_intensity = float(cv2.mean(smoothed, mask=mask)[0])
            peak_val = float(cv2.minMaxLoc(smoothed, mask=mask)[1])

            # Find the peak pixel within this contour
            masked_smoothed = np.where(mask > 0, smoothed, 0)
            peak_gy, peak_gx = np.unravel_index(np.argmax(masked_smoothed), smoothed.shape)

            # Enclosing circle for radius
            (cx, cy), radius_px = cv2.minEnclosingCircle(cnt)

            # Score: area fraction × peak-weighted mean
            area_norm = area / (h * w)
            score = area_norm * (0.5 * mean_intensity + 0.5 * peak_val)

            candidates.append({
                "score": score,
                "cx": cx, "cy": cy,
                "peak_gx": float(peak_gx), "peak_gy": float(peak_gy),
                "radius_px": radius_px,
                "intensity": mean_intensity,
                "peak": peak_val,
                "area": area,
                "area_norm": area_norm,
            })

        if not candidates:
            gy, gx = np.unravel_index(np.argmax(smoothed), smoothed.shape)
            return [_make_hotspot(gx / w, gy / h, float(smoothed[gy, gx]), 0.0, 1, (h, w), image_size)]

        # Sort descending by score
        candidates.sort(key=lambda x: x["score"], reverse=True)

        # ── 7. Non-Maximum Suppression — remove overlapping circles ──
        kept = []
        for cand in candidates:
            suppress = False
            for k in kept:
                dist = np.hypot(cand["cx"] - k["cx"], cand["cy"] - k["cy"])
                min_r = min(cand["radius_px"], k["radius_px"])
                max_r = max(cand["radius_px"], k["radius_px"])
                # Overlap when distance < sum of radii scaled by threshold
                if dist < (min_r + max_r) * nms_overlap_threshold:
                    suppress = True
                    break
            if not suppress:
                kept.append(cand)
            if len(kept) >= max_spots:
                break

        # ── 8. Build hotspot dicts ──
        hotspots = []
        for rank, cand in enumerate(kept, start=1):
            # Use peak pixel as the precise anchor point for pin placement
            hs = _make_hotspot(
                cx_norm=cand["peak_gx"] / w,
                cy_norm=cand["peak_gy"] / h,
                intensity=cand["intensity"],
                area_norm=cand["area_norm"],
                rank=rank,
                heatmap_shape=(h, w),
                image_size=image_size,
                radius_px=cand["radius_px"],
            )
            hotspots.append(hs)

        return hotspots

    def cleanup(self):
        for handler in self.handlers:
            handler.remove()


# ─── Hotspot Helpers ──────────────────────────────────────────────────────────

def _make_hotspot(
    cx_norm: float,
    cy_norm: float,
    intensity: float,
    area_norm: float,
    rank: int,
    heatmap_shape: tuple,
    image_size: Optional[tuple] = None,
    radius_px: float = 0.0,
) -> dict:
    h, w = heatmap_shape

    if image_size:
        iw, ih = image_size
        xPct = cx_norm * 100
        yPct = cy_norm * 100
        radius = (radius_px / w) * 100
    else:
        xPct = cx_norm * 100
        yPct = cy_norm * 100
        radius = (radius_px / w) * 100 if radius_px else 3.0

    xPct = float(np.clip(xPct, 4, 96))
    yPct = float(np.clip(yPct, 4, 96))

    return asdict(DiseaseHotspot(
        xPct=round(xPct, 2),
        yPct=round(yPct, 2),
        intensity=round(float(intensity), 4),
        radius=round(max(float(radius), 2.5), 2),
        rank=rank,
        label=_position_label(cx_norm, cy_norm),
        area_pct=round(float(area_norm) * 100, 2),
    ))


def _position_label(cx_norm: float, cy_norm: float) -> str:
    col = "left" if cx_norm < 0.35 else "right" if cx_norm > 0.65 else "centre"
    row = "upper" if cy_norm < 0.35 else "lower" if cy_norm > 0.65 else "mid"
    mapping = {
        ("upper", "centre"): "upper leaf tip",
        ("upper", "left"):   "upper-left margin",
        ("upper", "right"):  "upper-right margin",
        ("mid",   "centre"): "central leaf body",
        ("mid",   "left"):   "left-side margin",
        ("mid",   "right"):  "right-side margin",
        ("lower", "centre"): "lower central area",
        ("lower", "left"):   "lower-left lobe",
        ("lower", "right"):  "lower-right lobe",
    }
    return mapping.get((row, col), "leaf body")


# ─── Overlay Utilities ────────────────────────────────────────────────────────

def apply_heatmap_overlay(original_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.50) -> Image.Image:
    """
    Refined Grad-CAM++ overlay with full-image coverage and vibrant coloration,
    matching the aesthetic of high-fidelity agricultural diagnostic tools.
    
    This version removes the leaf-only masking to provide a broader context 
    of the model's global attention across the entire plant specimen.
    """
    img_np = np.array(original_img.convert("RGB"))
    h, w = img_np.shape[:2]

    # 1. Resize heatmap to image dimensions (using cubic for smoother/vibrant blocks)
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # 2. Thresholding + Soft Blur to create the 'glowing' block effect
    # This helps mimic the blocky yet vibrant look of lower-resolution feature maps
    heatmap_blurred = cv2.GaussianBlur(heatmap_resized, (15, 15), 0)
    heatmap_blurred = np.clip(heatmap_blurred, 0, 1)

    # 3. Apply high-contrast JET colormap
    heatmap_8bit = np.uint8(255 * heatmap_blurred)
    heatmap_color = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB).astype(np.float32)

    # 4. Global Blend (no masking)
    # Using a 0.5 alpha gives a strong indicator while keeping original details visible
    img_f = img_np.astype(np.float32)
    blended = img_f * (1 - alpha) + heatmap_color * alpha
    
    # Clip and convert back to uint8
    result = np.clip(blended, 0, 255).astype(np.uint8)
    
    return Image.fromarray(result)


def annotate_disease_on_original(
    original_img: Image.Image,
    hotspots: List[dict],
    heatmap: np.ndarray,
) -> Image.Image:
    """
    NEW: Draws disease spotting annotations directly on the original image.

    For each hotspot this draws:
      - A semi-transparent filled ellipse sized to the hotspot's radius
      - A solid colored border circle
      - A numbered badge in the corner

    The fill color uses the hotspot's intensity to pick from green→amber→red.
    This is designed to look like the high-precision disease spotting used in
    botanical research and professional agricultural diagnostic systems.
    """
    img_np = np.array(original_img.convert("RGB")).copy()
    h, w = img_np.shape[:2]

    # Build an overlay canvas for semi-transparent fills
    overlay = img_np.copy()

    def intensity_color(intensity: float) -> Tuple[int, int, int]:
        """Map 0→1 intensity to green→amber→red BGR."""
        if intensity >= 0.75:
            return (220, 38, 38)    # red — critical lesion
        elif intensity >= 0.55:
            return (234, 100, 14)   # orange — active zone
        elif intensity >= 0.35:
            return (234, 179, 8)    # amber — watch zone
        else:
            return (34, 197, 94)    # green — low activity

    for hs in hotspots:
        cx = int(hs["xPct"] / 100 * w)
        cy = int(hs["yPct"] / 100 * h)

        # Convert radius % → pixels; ensure minimum size
        r_px = max(int(hs["radius"] / 100 * w), 12)

        color = intensity_color(hs["intensity"])
        color_bgr = (color[2], color[1], color[0])  # PIL RGB → OpenCV BGR

        # ── Semi-transparent filled circle on overlay ──
        cv2.circle(overlay, (cx, cy), r_px, color_bgr, -1, cv2.LINE_AA)

        # Blend at 30% opacity for the fill
        img_np = cv2.addWeighted(overlay, 0.28, img_np, 0.72, 0)
        overlay = img_np.copy()  # reset overlay for next hotspot

        # ── Solid border ring ──
        cv2.circle(img_np, (cx, cy), r_px, color_bgr, 2, cv2.LINE_AA)

        # ── White outer glow ring ──
        cv2.circle(img_np, (cx, cy), r_px + 3, (255, 255, 255), 1, cv2.LINE_AA)

        # ── Rank badge (filled circle with number) ──
        badge_x = cx + r_px - 8
        badge_y = cy - r_px + 8
        cv2.circle(img_np, (badge_x, badge_y), 11, color_bgr, -1, cv2.LINE_AA)
        cv2.circle(img_np, (badge_x, badge_y), 11, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            img_np, str(hs["rank"]),
            (badge_x - 4, badge_y + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42,
            (255, 255, 255), 1, cv2.LINE_AA,
        )

        # ── Crosshair at exact peak ──
        cross_size = 5
        cv2.line(img_np, (cx - cross_size, cy), (cx + cross_size, cy), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(img_np, (cx, cy - cross_size), (cx, cy + cross_size), (255, 255, 255), 1, cv2.LINE_AA)

    return Image.fromarray(img_np)


def draw_hotspot_markers(
    overlay_img: Image.Image,
    hotspots: List[dict],
    heatmap_shape: tuple,
) -> Image.Image:
    """Server-side preview: numbered circles on heatmap overlay."""
    img_np = np.array(overlay_img.convert("RGB")).copy()
    h, w = img_np.shape[:2]

    for hs in hotspots:
        cx = int(hs["xPct"] / 100 * w)
        cy = int(hs["yPct"] / 100 * h)
        cv2.circle(img_np, (cx, cy), 18, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.circle(img_np, (cx, cy), 14, (220, 38, 38), -1, cv2.LINE_AA)
        cv2.putText(
            img_np, str(hs["rank"]),
            (cx - 4, cy + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (255, 255, 255), 2, cv2.LINE_AA,
        )
    return Image.fromarray(img_np)


def get_heatmap_base64(pil_img: Image.Image) -> str:
    import base64
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=87)
    return base64.b64encode(buf.getvalue()).decode("utf-8")