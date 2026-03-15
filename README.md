## FER Real-Time Reliability

### Install
pip install -r requirements.txt

### Offline (AffectNet subset)
python -m experiments.run_offline_affectnet --data_dir /path/to/affectnet_subset --epochs 3

### Real-time (webcam)
python -m experiments.run_realtime_scenarios --detector haar --smoothing none
python -m experiments.run_realtime_scenarios --detector retinaface --smoothing hybrid

Outputs:
- results/logs/*.csv : per-frame logs
- results/metrics/*.json : computed stability metrics
- results/gradcam/*.png : Grad-CAM snapshots (optional keypress)