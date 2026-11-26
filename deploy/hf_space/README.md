---
title: Roadside Traffic Intelligence
emoji: 🚦
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: agpl-3.0
short_description: Detect, track and count vehicles in traffic video
---

# Roadside Traffic Intelligence (DSCI-41)

Live demo of a KL University capstone project. Upload a short traffic video, place a counting line, and the app
detects vehicles (YOLO11n), tracks them (ByteTrack), counts line crossings by direction and class, flags difficult
conditions (night, blur, dense occlusion), and shows the results on an 8-page dashboard.

- **Source code and documentation:** https://github.com/Pavansailahar/Traffic-Volume-Estimation-and-Short-Term-Congestion-Forecasting-from-Roadside-Video
- **Limits:** CPU only; the first 60 seconds of each upload, downscaled to 1280 px wide. A 20-second clip takes about
  one to two minutes.
- **Privacy:** uploads are processed on this server and deleted when the Space restarts. Only upload footage you
  are allowed to share.
- **Status:** counts are estimates. Counting accuracy has not yet been validated against human reference counts.

This Space is deployed automatically from the GitHub repository (`deploy/hf_space/`). Do not edit it here: changes
are overwritten on the next deployment.
