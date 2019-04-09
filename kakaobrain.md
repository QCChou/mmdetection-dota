## Resources

- BrainCloud Image
  - mmdetect-190404

- mmcv : https://github.kakaocorp.com/kakaobrain/mmcv


## Train Cascaded RCNN Model

```
./tools/dist_train.sh configs/dota/cascade_rcnn_dconv_c3-c5_r50_fpn_1x_multi.py 4 \
    --validate \
    --work_dir work_dirs/test/
```
