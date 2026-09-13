package com.qujindai.localvideo;

import java.net.HttpURLConnection;
import java.util.concurrent.atomic.AtomicBoolean;

/** Unblocks a stalled model HTTP read when the UI requests cancellation. */
final class FaceDownloadCancellation implements AutoCloseable {
    private final Thread monitor;
    private volatile boolean closed;

    FaceDownloadCancellation(HttpURLConnection connection, AtomicBoolean cancelled) {
        monitor=new Thread(()->{
            while(!closed) {
                if(cancelled.get()) {
                    try {connection.disconnect();} catch(RuntimeException ignored) {}
                    // The worker may not have started HTTP yet. Keep disconnecting
                    // until close(), so an early cancel also catches a later connect.
                }
                try {Thread.sleep(80);} catch(InterruptedException finished) {return;}
            }
        },"face-download-cancel");
        monitor.setDaemon(true);
        monitor.start();
    }

    @Override public void close() {
        closed=true;
        monitor.interrupt();
        boolean interrupted=false;
        while(monitor.isAlive()) {
            try {monitor.join();} catch(InterruptedException issue) {interrupted=true;}
        }
        if(interrupted)Thread.currentThread().interrupt();
    }
}
