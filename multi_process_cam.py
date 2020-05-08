import cv2
import scipy.io as sio
import os
import numpy as np
import pygame as pg
import time
import multiprocessing as mp
import sys


def play_music(get_start, send_done):
    '''
    stream music with mixer.music module in a blocking manner
    this will stream the sound from disk while playing
    '''
    flg_start = False
    while True:
        print('Proc is running')
        try:
            flg_start = get_start.recv()
        except EOFError:
            print("Process #2 stop!")
            # send_stop.send(True)
            send_done.close()
            break

        if flg_start:
            print('set audio start')
            send_done.send(False)
            music_file = "service-bell_daniel_simion.mp3"
            # optional volume 0 to 1.0
            volume = 1
            # set up the mixer
            freq = 16000  # audio CD quality
            bitsize = -16  # unsigned 16 bit
            channels = 2  # 1 is mono, 2 is stereo
            buffer = 1024  # number of samples (experiment to get best sound)
            pg.mixer.init(freq, bitsize, channels, buffer)
            # volume value 0.0 to 1.0
            pg.mixer.music.set_volume(volume)
            clock = pg.time.Clock()
            try:
                pg.mixer.music.load(music_file)
                print("Music file {} loaded!".format(music_file))
            except pg.error:
                print("File {} not found! ({})".format(music_file, pg.get_error()))
                return
            pg.mixer.music.play()
            while pg.mixer.music.get_busy():
                # check if playback has finished
                clock.tick(30)
            print('Set audio done')
            send_done.send(True)


def motion_detection(cur_frame, pre_frame):
    h, w = cur_frame.shape[:2]
    d = cv2.absdiff(cur_frame, pre_frame)
    grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(grey, (3, 3), 0)

    diff_val = np.sum(grey) / (h * w)

    return grey, diff_val


def main():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    pre_frame = frame.copy()
    counter_init = 0

    mp.set_start_method('spawn')
    get_start, send_start = mp.Pipe(True)
    get_done, send_done = mp.Pipe(True)

    proc_2 = mp.Process(target=play_music, args=(get_start, send_done))
    proc_2.start()
    flg_done = True

    while True:
        ret, cur_frame = cap.read()
        if counter_init < 10:
            counter_init += 1
            pre_frame = cur_frame.copy()
            continue

        # processing code here
        blur, diff_val = motion_detection(cur_frame, pre_frame)
        pre_frame = cur_frame.copy()

        if get_done.poll():
            try:
                flg_done = get_done.recv()
                print('read done flag: ', flg_done)
            except EOFError:
                print("Process #2 stop!")
                # send_stop.send(True)
                # send_stop.close()
                break

        # print('Continue..')
        if diff_val >= 6 and flg_done:
            # play audio
            print(diff_val)
            send_start.send(True)

        # cv2.imshow('out', frame)
        cv2.imshow('out', blur)
        # Press Q on keyboard to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    get_start.close()
    send_start.close()
    get_done.close()
    send_done.close()
    proc_2.join()

    cap.release()


if __name__ == "__main__":
    sys.exit(main())
