#!/usr/bin/python
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import sys


# create coco object and cocoRes object
coco = COCO(sys.argv[1])
cocoRes = coco.loadRes(sys.argv[2])
model_name = sys.argv[3]
model = sys.argv[4]

# create cocoEval object by taking coco and cocoRes
cocoEval = COCOEvalCap(coco, cocoRes)
# evaluate on a subset of images by setting
cocoEval.params['image_id'] = cocoRes.getImgIds()

# evaluate results
cocoEval.evaluate()
# print output evaluation scores
email_msg = ""
for metric, score in cocoEval.eval.items():
    print '%s: %.3f' % (metric, score)
    email_msg += '%s: %.3f\n' % (metric, score)
for key, value in cocoEval.imgToEval.iteritems():
    print key, value

