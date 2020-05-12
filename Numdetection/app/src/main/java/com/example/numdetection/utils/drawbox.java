package com.example.numdetection.utils;

import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PorterDuff;
import android.graphics.Rect;

import java.util.List;

public class drawbox {

    /**
     * 在给定的canvas上，绘制面部识别点以及识别框
     *
     * @param canvas canvas
     */
    public static void drawBlazeFacePoint(List<Boxpoint> lbp, Canvas canvas) {
        if (lbp == null) {
            return;
        }
        Paint paint = new Paint();
        paint.setStrokeWidth(2.f);
        paint.setStyle(Paint.Style.FILL);
        paint.setColor(Color.RED);
        //刷新画布
        canvas.drawColor(Color.BLACK, PorterDuff.Mode.CLEAR);
//        int screenWidth = canvas.getWidth();
//        int screenHeight = canvas.getHeight();
//        int left = 0;
//        int top = 0;
//        float widthRatio = (float) screenWidth / AIConstants.MODEL2_WIDTH;
//        float heightRatio = (float) screenHeight / AIConstants.MODEL2_HEIGHT;

        for(int i = 0;i<lbp.size();i++) {
            Boxpoint bp = lbp.get(i);
            //绘制识别框
            Rect rect = new Rect(bp.fpoint.x, bp.fpoint.y, bp.spoint.x, bp.spoint.y);
            paint.setStyle(Paint.Style.STROKE);
            canvas.drawRect(rect, paint);
            canvas.drawText(bp.string, bp.fpoint.x, bp.fpoint.y, paint);
        }
    }
}
