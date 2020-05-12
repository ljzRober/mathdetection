package com.example.numdetection.utils;

import org.opencv.core.Mat;

import java.io.BufferedReader;
import java.io.InputStreamReader;

import android.content.Context;

import static org.opencv.core.CvType.CV_16S;
import static org.opencv.core.CvType.CV_32F;
import static org.opencv.core.CvType.CV_32S;

public class nn {

    public static String nnop(Mat mat, Context context){
        String path1 = "theta1.csv";
        String path2 = "theta2.csv";
        String path3 = "theta3.csv";

        Mat theta1 = new Mat(1000, 2026, CV_32F);
        for(int i = 0;i<1000;i++) {
            for (int j = 0; j < 2026; j++) {
                theta1.put(i, j, readcsv(context, path1)[i][j]);
            }
        }
        Mat theta2 = new Mat(500, 1001, CV_32F);
        for(int i = 0;i<500;i++) {
            for (int j = 0; j < 1001; j++) {
                theta2.put(i, j, readcsv(context, path2)[i][j]);
            }
        }
        Mat theta3 = new Mat(15, 501, CV_32F);
        for(int i = 0;i<1000;i++) {
            for (int j = 0; j < 2026; j++) {
                theta3.put(i, j, readcsv(context, path3)[i][j]);
            }
        }

        mat = mat.reshape(0, 1);
        Mat q = new Mat(0, 2026, CV_32S);
        for(int i = 0;i<2025;i++) {
            if(i == 0){q.put(0, i, 1);}
            q.put(0, i, mat.get(0, i+1));
        }
        Mat z2 = multiply(theta1.t(), q);
        z2 = sigmoid(z2);
        Mat a2 = new Mat(0, 1001, CV_32S);
        for(int i = 0;i<1001;i++) {
            if(i == 0){q.put(0, i, 1);}
            a2.put(0, i, z2.get(0, i+1));
        }
        Mat z3 = multiply(theta2.t(), a2);
        z3 = sigmoid(z3);
        Mat a3 = new Mat(0, 501, CV_32S);
        for(int i = 0;i<1001;i++) {
            if(i == 0){q.put(0, i, 1);}
            a3.put(0, i, z3.get(0, i+1));
        }
        Mat z4 = multiply(theta3.t(), a3);
        Mat a4 = sigmoid(z4);
        String[] words = new String[]{"+", "-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "=", "div", "x"};
        double d = 0.0;
        int select = 0;
        for(int m = 0;m <a4.cols();m++){
            if(a4.get(0, m)[0]>d){
                select = m;
            }
        }
        System.out.println("数字为："+words[select]);
        return words[select];
    }

    private static Mat sigmoid(Mat mat){
        for(int i = 0;i<1000;i++) {
            for (int j = 0; j < 2026; j++) {
                double[] f = mat.get(i, j);
                mat.put(i, j, 1/(1 + Math.exp(f[0])));
            }
        }
        return mat;
    }

    private static Mat multiply(Mat mat1, Mat mat2){
        Mat result = new Mat(1, mat2.cols(), CV_32F);
        for(int i = 0;i<mat2.cols();i++){
            double f = 0.0;
            for(int j = 0; j<mat1.cols(); j++) {
                f += mat1.get(0, j)[0] * mat2.get(j, i)[0];
            }
            result.put(0, i, f);
        }
        return result;
    }

    public static float[][] readcsv(Context context, String name){
            String line;
            float[][] arry = new float[1000][2026];
            if("theta2.csv".equals(name)) {
                arry = new float[500][1001];
            }
            else if("theta3.csv".equals(name)){
                arry = new float[15][501];
            }
            try {
                BufferedReader reader =
                        new BufferedReader(new InputStreamReader(context.getAssets().open(name)));
                int count = 0;
                while((line=reader.readLine())!=null){
                    count++;
                    if(count == 0){continue;}
                    if(count == arry.length+1){break;}
                    String item[] = line.split(",");//CSV格式文件为逗号分隔符文件，这里根据逗号切分
                    for (int i = 0;i<item.length-1; i++) {
                        String last = item[i+1];//这就是你要的数据了
                        arry[count-1][i] = Float.parseFloat(last);//如果是数值，可以转化为数值
                    }
                }
                reader.close();
                return arry;
            } catch (Exception e) {
                e.printStackTrace();
            }
            return arry;
    }

}
