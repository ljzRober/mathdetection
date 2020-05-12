package com.example.numdetection.utils;
import android.graphics.Rect;

public class AIConstants {
    public static final String TAG = "SNPoseEstimation";
    public static final boolean isDebug = true;
    public Rect rect = new Rect();

    //nn模型，bitmap的宽、高
    public static final int MODEL2_WIDTH = 45;
    public static final int MODEL2_HEIGHT = 45;

    //-------------- 数字识别 部分常量 -----------------------//
    public final static String BF_MODEL_PATH = "face_detection_front.tflite";
}
