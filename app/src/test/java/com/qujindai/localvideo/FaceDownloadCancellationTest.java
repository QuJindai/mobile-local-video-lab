package com.qujindai.localvideo;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.FutureTask;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.Test;

public class FaceDownloadCancellationTest {
    @Test public void cancellationDisconnectsAStalledRead() throws Exception {
        CountDownLatch reading=new CountDownLatch(1),disconnected=new CountDownLatch(1);
        HttpURLConnection connection=new HttpURLConnection(new URL("https://example.invalid/model")) {
            @Override public void disconnect(){disconnected.countDown();}
            @Override public boolean usingProxy(){return false;}
            @Override public void connect(){}
            @Override public InputStream getInputStream(){return new InputStream(){
                @Override public int read() throws IOException {
                    reading.countDown();
                    try {
                        if(!disconnected.await(3,TimeUnit.SECONDS))throw new IOException("not disconnected");
                    } catch(InterruptedException e){throw new IOException(e);}
                    throw new IOException("cancelled connection");
                }
            };}
        };
        AtomicBoolean cancelled=new AtomicBoolean();
        FutureTask<Boolean> blocked=new FutureTask<>(()->{
            try {connection.getInputStream().read();return false;}
            catch(IOException e){return e.getMessage().equals("cancelled connection");}
        });
        Thread reader=new Thread(blocked,"test-model-read");
        reader.setDaemon(true);
        try(FaceDownloadCancellation monitor=new FaceDownloadCancellation(connection,cancelled)) {
            reader.start();
            assertTrue(reading.await(1,TimeUnit.SECONDS));
            cancelled.set(true);
            assertTrue("Cancellation must disconnect instead of waiting for HTTP timeout",disconnected.await(1,TimeUnit.SECONDS));
            assertTrue(blocked.get(1,TimeUnit.SECONDS));
        } finally {connection.disconnect();reader.interrupt();}
    }

    @Test public void cancellationBeforeConnectStillDisconnectsAfterConnect() throws Exception {
        CountDownLatch firstDisconnect=new CountDownLatch(1),afterConnect=new CountDownLatch(1);
        AtomicBoolean connected=new AtomicBoolean();
        HttpURLConnection connection=new HttpURLConnection(new URL("https://example.invalid/model")) {
            @Override public void disconnect(){
                firstDisconnect.countDown();
                if(connected.get())afterConnect.countDown();
            }
            @Override public boolean usingProxy(){return false;}
            @Override public void connect(){connected.set(true);}
        };
        try(FaceDownloadCancellation monitor=new FaceDownloadCancellation(connection,new AtomicBoolean(true))) {
            assertTrue(firstDisconnect.await(1,TimeUnit.SECONDS));
            connection.connect();
            assertTrue("An early cancel must cover a later HTTP start",afterConnect.await(1,TimeUnit.SECONDS));
        }
    }

    @Test public void closePreservesCallerInterrupt() throws Exception {
        HttpURLConnection connection=new HttpURLConnection(new URL("https://example.invalid/model")) {
            @Override public void disconnect(){fail("Completed request must not be cancelled");}
            @Override public boolean usingProxy(){return false;}
            @Override public void connect(){}
        };
        FaceDownloadCancellation monitor=new FaceDownloadCancellation(connection,new AtomicBoolean());
        try {
            Thread.currentThread().interrupt();
            monitor.close();
            assertTrue(Thread.currentThread().isInterrupted());
        } finally {Thread.interrupted();}
    }
}
