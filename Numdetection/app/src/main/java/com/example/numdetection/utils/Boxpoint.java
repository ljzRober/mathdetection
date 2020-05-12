package com.example.numdetection.utils;

import android.graphics.Point;

public class Boxpoint {

    public Point fpoint;
    public Point spoint;
    public String string;

    public String getString() {
        return string;
    }

    public void setString(String string) {
        this.string = string;
    }

    public Point getFpoint() {
        return fpoint;
    }

    public Point getSpoint() {
        return spoint;
    }

    public void setSpoint(Point spoint) {
        this.spoint = spoint;
    }

    public void setFpoint(Point fpoint) {
        this.fpoint = fpoint;
    }


    public Boxpoint(Point fpoint, Point spoint){
        this.fpoint = fpoint;
        this.spoint = spoint;
    }
    public Boxpoint(Point fpoint, Point spoint, String string){
        this.fpoint = fpoint;
        this.spoint = spoint;
        this.string = string;
    }
}
