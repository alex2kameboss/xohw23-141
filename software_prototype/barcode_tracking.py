# -> bbox -> barcode_bbox

import numpy as np

from sort.sort import Sort

# create instance of SORT
mot_tracker = Sort(max_age=1, min_hits=3, iou_threshold=0.1)
directions = {}
ignore = set()


def barcode_tracking(bboxes):
    if len(bboxes) == 0:
        track_bbs_ids = mot_tracker.update()
    else:
        detections = np.array(bboxes)
        track_bbs_ids = mot_tracker.update(detections)
    #print("tracking:", track_bbs_ids)

    # update directions
    for track in track_bbs_ids:
        id = int(track[4])
        if id in directions:
            dir = 'l'
            if track[2] - directions[id][0][2] >= 0:
                dir = 'r'
            directions[id] = (track[0:4], dir)
        else:
            directions[id] = (track[0:4], 'r')

    #print(directions)

    # remove directions
    current = set(np.array(track_bbs_ids[:, 4], dtype=int))
    old = set(directions.keys())
    remove = old.difference(current)
    #print(remove)
    for key in remove:
        del directions[key]

    # process output
    ret = []
    old = set(directions.keys())
    analyse = old.difference(ignore)
    #print("analyse:", analyse)
    for key in analyse:
        if directions[key][0][0] <= 50:
            ret.append(directions[key][0])
            ignore.add(key)

    #print("directions:", directions)
    return ret


if __name__ == '__main__':
    coords1 = [10, 10, 50, 50]
    coords2 = [1000, 1000, 1050, 1050]
    dirs = [10, 2]
    i = 0

    while True:
        vals = barcode_tracking([[coords1[0] + i * dirs[0], coords1[1] + i * dirs[1],
                                  coords1[2] + i * dirs[0], coords1[3] + i * dirs[1]],
                                 [coords2[0] - i * dirs[0], coords2[1] - i * dirs[1],
                                  coords2[2] - i * dirs[0], coords2[3] - i * dirs[1]]
                                 ])

        if coords1[2] + i * dirs[0] >= 640:
            break

        i += 1
        print(coords1[2] + i * dirs[0], vals)
