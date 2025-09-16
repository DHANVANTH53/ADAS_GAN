# inference.py
import cv2
import torch
import numpy as np
from models.unet import UNet
import argparse
import time

def preprocess_frame(frame, img_size=256):
    # BGR->RGB, resize, normalize to [0,1], to tensor
    import torchvision.transforms as T
    t = T.Compose([T.ToPILImage(), T.Resize((img_size, img_size)), T.ToTensor()])
    return t(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0)  # 1,C,H,W

def tensor_to_image(tensor):
    # tensor: 1,C,H,W in [0,1]
    img = tensor.squeeze(0).permute(1,2,0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def run_webcam(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    cap = cv2.VideoCapture(0 if args.cam == 0 else args.cam)
    fps_time = 0

    # Optional: load YOLO model (ultralytics)
    yolo = None
    if args.yolo:
        try:
            from ultralytics import YOLO
            yolo = YOLO(args.yolo_model)  # e.g., 'yolov8n.pt'
        except Exception as e:
            print("YOLO not available or failed to load:", e)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h0, w0 = frame.shape[:2]
        # preprocess
        inp = preprocess_frame(frame, img_size=args.img_size)
        inp = inp.to(device)
        with torch.no_grad():
            out = model(inp)
        out_img = tensor_to_image(out)
        # Resize restored back to original size for display
        out_img = cv2.resize(out_img, (w0, h0))

        # Optionally run detector on restored frame
        if yolo:
            results = yolo.predict(source=out_img, save=False, imgsz=640, conf=0.35, verbose=False)
            # results[0].boxes show boxes, draw them
            det = results[0]
            if len(det.boxes) > 0:
                for box in det.boxes:
                    x1,y1,x2,y2 = map(int, box.xyxy[0].cpu().tolist())
                    conf = float(box.conf[0].cpu().numpy()) if hasattr(box, 'conf') else 0.0
                    cls = int(box.cls[0].cpu().numpy()) if hasattr(box, 'cls') else 0
                    cv2.rectangle(out_img, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(out_img, f"{cls}:{conf:.2f}", (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Show before (left) and after (right)
        combined = np.concatenate([frame, out_img], axis=1)
        # FPS overlay
        curr_time = time.time()
        fps = 1.0/(curr_time - fps_time + 1e-6)
        fps_time = curr_time
        cv2.putText(combined, f"FPS: {fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)

        cv2.imshow('Before | After (press q to exit)', combined)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='checkpoints/best_ckpt.pth')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--cam', type=int, default=0)
    parser.add_argument('--yolo', action='store_true', help='enable YOLO detection overlay on restored frames')
    parser.add_argument('--yolo_model', type=str, default='yolov8n.pt')
    args = parser.parse_args()
    run_webcam(args)
 
