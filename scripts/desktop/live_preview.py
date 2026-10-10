"""Original bounded Tkinter camera preview; no recordings."""
import collections
import json
import queue
import threading
import time
import cv2

def put_latest(mailbox, value):
    try:
        mailbox.put_nowait(value)
    except queue.Full:
        try:
            mailbox.get_nowait()
        except queue.Empty:
            pass
        mailbox.put_nowait(value)

def live(model, camera, window_seconds, update_seconds, duration_seconds):
    import tkinter as tk
    from PIL import Image, ImageTk
    fps = model.protocol['target_fps']
    n = int(round(window_seconds*fps))
    model.warmup(n)
    stop = threading.Event()
    lock = threading.Lock()
    buffer = collections.deque(maxlen=n)
    state = {'preview':None,'result':None,'error':None,'captured':0,'sampled':0,'inferred':0,'displayed':0}
    mailbox = queue.Queue(maxsize=1)
    cap = cv2.VideoCapture(camera,cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open camera {camera}; close other camera apps or use --camera 1.')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

    def capture():
        next_sample = 0.
        try:
            while not stop.is_set():
                ok, frame = cap.read()
                stamp = time.perf_counter()
                if not ok:
                    raise RuntimeError('Camera stopped returning frames.')
                with lock:
                    state['preview'] = frame
                    state['captured'] += 1
                    if next_sample == 0:
                        next_sample = stamp
                    if stamp >= next_sample:
                        buffer.append((stamp,frame.copy()))
                        state['sampled'] += 1
                        # No catch-up duplicates if capture stalls.
                        next_sample += (int((stamp-next_sample)*fps)+1)/fps
        except Exception as error:
            with lock:state['error'] = str(error)
            stop.set()

    def inference():
        try:
            while not stop.is_set():
                try:
                    timestamps, frames = mailbox.get(timeout=.2)
                except queue.Empty:
                    continue
                result,_ = model.classify(frames)
                result['window_start'] = timestamps[0]
                result['window_end'] = timestamps[-1]
                result['completed'] = time.perf_counter()
                result['effective_sample_fps'] = (len(timestamps)-1)/(timestamps[-1]-timestamps[0])
                with lock:
                    state['result'] = result
                    state['inferred'] += 1
                    count = state['inferred']
                print('LIVE_PREDICTION',json.dumps({'number':count,'label':result['top3'][0]['label'],
                    'score':result['top3'][0]['score'],'processing_ms':result['processing_ms'],
                    'effective_sample_fps':result['effective_sample_fps']}),flush=True)
        except Exception as error:
            with lock:state['error'] = str(error)
            stop.set()

    readers = [threading.Thread(target=capture,daemon=True),threading.Thread(target=inference,daemon=True)]
    title = 'KARD live HAR - Q/Escape to quit'
    print('CAMERA_OPEN', json.dumps({'camera':camera,'window_seconds':window_seconds,
        'target_fps':fps,'update_seconds':update_seconds,'video_recorded':False}),flush=True)
    started = time.perf_counter()
    viewer = None
    try:
        # xai uses headless OpenCV; Tkinter displays without changing its packages.
        viewer = tk.Tk()
        viewer.title(title)
        viewer.geometry('960x720')
        viewer.resizable(False,False)
        viewer.protocol('WM_DELETE_WINDOW',stop.set)
        viewer.bind('<Escape>',lambda event:stop.set())
        viewer.bind('<q>',lambda event:stop.set())
        viewer.bind('<Q>',lambda event:stop.set())
        display = tk.Label(viewer,bg='#0F141C',borderwidth=0)
        display.pack(fill='both',expand=True)
        viewer.update()
        for thread in readers:thread.start()
        next_update = started
        while not stop.is_set():
            now = time.perf_counter()
            if duration_seconds and now-started >= duration_seconds:
                break
            with lock:
                frame = None if state['preview'] is None else state['preview'].copy()
                result = state['result']
                count = len(buffer)
                snapshot = list(buffer) if count==n and now>=next_update else None
            if snapshot:
                put_latest(mailbox,([p[0] for p in snapshot],[p[1] for p in snapshot]))
                next_update = now+update_seconds
            if frame is None:
                viewer.update()
                time.sleep(.01)
                continue
            frame = cv2.resize(frame,(960,720))
            panel = frame.copy()
            cv2.rectangle(panel,(0,0),(960,175),(15,20,28),-1)
            frame = cv2.addWeighted(panel,.85,frame,.15,0)
            lines = [f'KARD A0 | {model.runtime_name} | {fps:g} fps | {window_seconds:g}s window | Q/Esc: quit']
            if result:
                age = max(0,now-result['window_end'])
                lines += [f"{rank+1}. {p['label']}   score {100*p['score']:.1f}%" for rank,p in enumerate(result['top3'])]
                lines += [f"Inference {result['processing_ms']:.0f} ms | result age {age:.1f}s | sampled {result['effective_sample_fps']:.1f} fps"]
            else:
                lines += [f'Collecting temporal context: {count}/{n} frames']
            for i,line in enumerate(lines):
                cv2.putText(frame,line,(16,28+i*30),cv2.FONT_HERSHEY_SIMPLEX,.65,(235,242,247),1,cv2.LINE_AA)
            cv2.rectangle(frame,(0,673),(960,720),(15,20,28),-1)
            cv2.putText(frame,'18 KARD actions; idle/unknown not trained. Scores are uncalibrated.',(14,702),
                        cv2.FONT_HERSHEY_SIMPLEX,.58,(220,225,235),1,cv2.LINE_AA)
            picture = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))
            display.configure(image=picture)
            display.image = picture
            viewer.update()
            state['displayed'] += 1
            time.sleep(.01)
    finally:
        stop.set()
        for thread in readers:
            if thread.is_alive():thread.join(timeout=2)
        cap.release()
        if viewer is not None:
            try:viewer.destroy()
            except tk.TclError:pass
    if state['error']:
        raise RuntimeError(state['error'])
    print('LIVE_STOPPED',json.dumps({'captured':state['captured'],'sampled':state['sampled'],
                                   'displayed':state['displayed'],'predictions':state['inferred'],'video_recorded':False}),flush=True)
