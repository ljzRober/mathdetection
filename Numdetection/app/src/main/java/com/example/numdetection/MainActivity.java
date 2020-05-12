package com.example.numdetection;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;
import android.view.Gravity;
import android.view.WindowManager;
import android.widget.Toast;

import androidx.annotation.Nullable;

import com.example.numdetection.R;
import com.example.numdetection.Camera2;
import com.example.numdetection.ui.AutoFitTextureView;
import com.example.numdetection.ui.SkeletonTextureView;
import com.example.numdetection.utils.AIConstants;
import com.example.numdetection.utils.AIImageUtils;
import com.example.numdetection.utils.Boxpoint;
import com.example.numdetection.utils.drawbox;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;

import java.util.List;

/**
 * 面部侦测
 * 面部框以及五官
 * 基于BlazeFace模型
 */
public class MainActivity extends Activity {
    private AutoFitTextureView mAutoFitTextureView;
    private SkeletonTextureView mSkeletonTextureView;
    private Camera2 mCamera2 = new Camera2();

    private Runnable runnable = new Runnable() {
        @Override
        public void run() {
            synchronized (this) {
                if (mCamera2.isRunning()) {
                    classifyFrame();
                }
            }
            mCamera2.poseTask(runnable);
        }
    };

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        initView();
    }

    //OpenCV库加载并初始化成功后的回调函数
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {

        @Override
        public void onManagerConnected(int status) {
            // TODO Auto-generated method stub
            switch (status){
                case BaseLoaderCallback.SUCCESS:
                    Log.i(AIConstants.TAG, "成功加载");
                    Toast toast = Toast.makeText(getApplicationContext(),
                            "成功加载！", Toast.LENGTH_LONG);
                    toast.setGravity(Gravity.CENTER, 0, 0);
                    toast.show();
                    break;
                default:
                    super.onManagerConnected(status);
                    Log.i(AIConstants.TAG, "加载失败");
                    Toast toast1 = Toast.makeText(getApplicationContext(),
                            "加载失败！", Toast.LENGTH_LONG);
                    toast1.setGravity(Gravity.CENTER, 0, 0);
                    toast1.show();
                    break;
            }

        }
    };

    @Override
    protected void onResume() {
        super.onResume();
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        mCamera2.onResume(this, true);
        mCamera2.setTextureView(mAutoFitTextureView);
        mCamera2.setSkeletonTextureView(mSkeletonTextureView);
        mCamera2.poseTask(runnable);
        if (!OpenCVLoader.initDebug()) {
            Log.d(AIConstants.TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(AIConstants.TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    protected void onPause() {
        mCamera2.onPause();
        getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
    }

    private void initView() {
        mAutoFitTextureView = (AutoFitTextureView) findViewById(R.id.sfv1);
        mSkeletonTextureView = (SkeletonTextureView) findViewById(R.id.stv1);
        mSkeletonTextureView.setOpaque(false);
    }

    private void classifyFrame() {
        if (!mCamera2.isRunning()) {
            Log.e(AIConstants.TAG, "没有初始化完成或者摄像头没有工作!");
            return;
        }
        if (!mAutoFitTextureView.isAvailable()) {
            Log.e(AIConstants.TAG, "mTextureView不可用!");
            return;
        }
        Bitmap bitmap = null;
        Bitmap resizeBitmap = null;
        try {
            bitmap = mAutoFitTextureView.getBitmap();
            Size size = new Size(bitmap.getWidth(), bitmap.getHeight());
            List<Boxpoint> lbp = AIImageUtils.waijiematrix(bitmap, this);
            //resizeBitmap = AIImageUtils.cropSkeletonBitmap(cropBitmap, 128, 128);
            if (lbp == null) {
                return;
            }
            Canvas canvas = mSkeletonTextureView.lockCanvas();
            if (canvas != null) {
                for(int i = 0; i<lbp.size(); i++)
                drawbox.drawBlazeFacePoint(lbp, canvas);
                mSkeletonTextureView.unlockCanvasAndPost(canvas);
            }
        } catch (Exception ex) {
            ex.printStackTrace();
        } finally {
            if (bitmap != null) {
                bitmap.recycle();
            }
            if (resizeBitmap != null) {
                resizeBitmap.recycle();
            }
        }
    }
}
