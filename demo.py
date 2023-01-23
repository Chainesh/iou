import argparse
import numpy as np
from iou_tracker import track_iou
from util import load_mot, save_to_csv
import cv2


def main(args):

    nms = None
    sigma_l = 0
    sigma_h = 0.5
    sigma_iou = 0.5
    t_min = 2
    format_ = "motchallenge" 
    

    detections = load_mot(args.detection_path, nms_overlap_thresh=nms, with_classes=False)
    # detections = load_mot(arr, nms_overlap_thresh=nms, with_classes=False)
    tracks = track_iou(detections, sigma_l, sigma_h, sigma_iou, t_min)
    save_to_csv(args.output_path, tracks, fmt=format_)

    height, width  = 640 , 480
    black_img = np.zeros((height,width,3), np.uint8)
    for i, frame in enumerate(detections):
        # for detection in frame:
        #     print(detection)
        #     for bbox in detection['bbox']:
        #     print(int(bbox[0]), int(bbox[1]),int(bbox[2]), int(bbox[3]))               
        #         cv2.rectangle(black_img, (int(bbox[2]), int(bbox[3])), (int(bbox[4]), int(bbox[5])), (0, 0, 255), 2)
        for track in tracks:
            for bbox in track['bboxes']:
                cv2.rectangle(black_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
        cv2.imshow("Tracking", black_img)
        cv2.waitKey(0)
        cv2.imwrite("track_"+str(i)+".jpg", black_img) # save the image
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="IOU Tracker")
    parser.add_argument('-d', '--detection_path', type=str, required=True, help="full path to CSV file containing the detections")
    parser.add_argument('-o', '--output_path', type=str, required=True, help="output path to store the tracking results (MOT challenge)")

    args = parser.parse_args()
    main(args)


