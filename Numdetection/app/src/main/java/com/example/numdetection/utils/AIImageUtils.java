package com.example.numdetection.utils;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Point;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import static org.opencv.core.CvType.CV_16S;
import static org.opencv.core.CvType.CV_32S;
import static org.opencv.imgproc.Imgproc.INTER_CUBIC;
import static org.opencv.imgproc.Imgproc.THRESH_BINARY;
import static org.opencv.imgproc.Imgproc.boundingRect;
import static org.opencv.imgproc.Imgproc.findContours;
import static org.opencv.imgproc.Imgproc.resize;
import static org.opencv.imgproc.Imgproc.threshold;

/**
 * 图像处理部分
 */
public class AIImageUtils {

    public static List<Boxpoint> waijiematrix(Bitmap bitmap, Context context){
        Mat rgbMat = new Mat();
        Mat grayMat = new Mat();
        Mat binaryMat = new Mat();
        Utils.bitmapToMat(bitmap, rgbMat);//convert original bitmap to Mat, R G B.
        Imgproc.cvtColor(rgbMat, grayMat, Imgproc.COLOR_RGB2GRAY);//rgbMat to gray grayMat
        threshold(grayMat, binaryMat, 125, 255, THRESH_BINARY);
        Log.i(AIConstants.TAG, "procSrc2Gray sucess...");

        List<MatOfPoint> contours = new ArrayList<>();
        List<Boxpoint> pointset= new ArrayList<>();
        Mat hierarchy = new Mat();
        Rect rect = new Rect();
        findContours(binaryMat, contours, hierarchy, Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_SIMPLE);
        System.out.println("轮廓数量："+ contours.size());
        System.out.println("hierarchy类型："+ hierarchy);
        int num = 0;
        for(Iterator<MatOfPoint> it = contours.iterator(); it.hasNext();){
            num++;
            MatOfPoint matOfPoint = it.next();
            rect = boundingRect(matOfPoint);
            double size1 = rect.width;
            double size2 = rect.height;
            if(size1>AIConstants.MODEL2_HEIGHT && size2>AIConstants.MODEL2_WIDTH){
                Point p1 = new Point(new Double(rect.tl().x).intValue(), new Double(rect.tl().y).intValue());
                Point p2 = new Point(new Double(rect.br().x).intValue(), new Double(rect.br().y).intValue());

                Size size = new Size(AIConstants.MODEL2_WIDTH, AIConstants.MODEL2_HEIGHT);
                Mat remat = new Mat(size, CV_32S);
                Mat cropimg = binaryMat.submat(rect);
                resize(cropimg, remat, size);
                String s = nn.nnop(remat, context);
                pointset.add(new Boxpoint(p1, p2, s));
            }
        }
        return pointset;
    }

}
