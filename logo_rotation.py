"""
Independent Simulation: Continuously spins a text layer horizontally 
in 3D space (Y-axis rotation) using OpenCV perspective transformations.
"""

import cv2
import numpy as np

def create_text_layer(text, font, scale, color, thickness):
    """ Renders text onto a square bounding box. """
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    #box_size = max(text_w, text_h) + 60  # Extra padding prevents 3D corner clipping
    box_size = max(text_w, text_h)
    
    layer = np.zeros((box_size-70, text_w + 10, 3), dtype=np.uint8)
    # Center text horizontally and vertically inside the box
    x = (box_size - text_w + 10) // 2
    y = (box_size - 70 + text_h) // 2 - (baseline // 2)
    
    cv2.putText(layer, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    return layer

def get_horizontally_rotated_layer(image, angle_degrees, fov=200):
    """
    Projects a 2D image into 3D space, rotates it around the Y-axis 
    (horizontal coin spin), and maps it back to a 2D frame via perspective projection.
    """
    h, w = image.shape[:2]
    rad = np.deg2rad(angle_degrees)
    
    # 1. Define original 2D corner points of the source image
    src_pts = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype=np.float32)
    
    # 2. Compute 3D coordinates centered at (0,0,0) before rotation
    cx, cy = w / 2.0, h / 2.0
    pts_3d = np.array([
        [-cx, -cy, 0],
        [ cx, -cy, 0],
        [ cx,  cy, 0],
        [-cx,  cy, 0]
    ], dtype=np.float32)
    
    # 3. Apply 3D Rotation Matrix around the Y-Axis (Horizontal Rotation)
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)
    
    rotated_pts_3d = np.zeros_like(pts_3d)
    for i in range(4):
        x3d, y3d, z3d = pts_3d[i]
        # Y coordinates stay completely identical during a horizontal spin
        rotated_pts_3d[i][0] = x3d * cos_a + z3d * sin_a  # X transform
        rotated_pts_3d[i][1] = y3d                        # Y stays same
        rotated_pts_3d[i][2] = -x3d * sin_a + z3d * cos_a # Z transform (Depth)
        
    # 4. Project the transformed 3D points back onto a 2D perspective plane
    dst_pts = np.zeros((4, 2), dtype=np.float32)
    for i in range(4):
        x3d, y3d, z3d = rotated_pts_3d[i]
        # Perspective division factor handles the '3D scaling look' as edges get closer/further
        distance_factor = fov / (fov + z3d) 
        dst_pts[i][0] = cx + x3d * distance_factor
        dst_pts[i][1] = cy + y3d * distance_factor

    # 5. Calculate warp perspective matrix and transform image mapping
    perspective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_image = cv2.warpPerspective(
        image, 
        perspective_matrix, 
        (w, h), 
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=(0, 0, 0)
    )
    return warped_image

# --- SETUP VARIABLES ---
text_string = "JARVIS"
font_style = cv2.FONT_HERSHEY_COMPLEX
font_scale = 1.3
text_color = (212, 82, 0)  # Green text
thickness = 4

# Generate the initial flat text layer template
text_layer = create_text_layer(text_string, font_style, font_scale, text_color, thickness)

window_name = "3D Horizontal Spin Simulation"
#cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
spin_angle = 0

print("Starting 3D horizontal rotation loop. Press 'ESC' to exit.")

if __name__ == "__main__":
    # --- MAIN ANIMATION LOOP ---
    while True:
        # Build clean background canvas (500x500 pixels)
        canvas = np.zeros((500, 500, 3), dtype=np.uint8)
        # Build clean white-background canvas
        canvas = np.full((500, 500, 3), 255, dtype=np.uint8)
        
        # Progress spinning angle step (Increase to spin faster, decrease to slow down)
        spin_angle = (spin_angle + 4) % 360
        
        # Process current horizontal tilt layer frame (fov=250 provides clean depth perspective)
        rotated_text = get_horizontally_rotated_layer(text_layer, spin_angle, fov=250)
        th, tw = rotated_text.shape[:2]

        # Calculate offsets to draw the asset directly in the center of the canvas
        ch, cw = canvas.shape[:2]
        x_off = (cw - tw) // 2
        y_off = (ch - th) // 2
        
        # Isolate the corresponding Region of Interest (ROI) on the canvas
        roi = canvas[y_off:y_off+th, x_off:x_off+tw]
        
        # Use thresholds to isolate the text pixels from the empty rotated black background corners
        gray = cv2.cvtColor(rotated_text, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        
        # Segment out foreground and background, then combine them seamlessly
        bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        fg = cv2.bitwise_and(rotated_text, rotated_text, mask=mask)
        
        # Save the blended result back to the canvas sheet
        canvas[y_off:y_off+th, x_off:x_off+tw] = cv2.add(bg, fg)
        
        # Push matrix to rendering display
        cv2.imshow(window_name, canvas)
        
        # Break loop if user presses ESC key
        if cv2.waitKey(20) & 0xFF == 27:
            break

    cv2.destroyAllWindows()