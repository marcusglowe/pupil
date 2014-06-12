'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2013  Moritz Kassner & William Patera

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import os
import multiprocessing
import threading
import cv2
import cv
import zmq
import time
from image_outline import ImageOutline
from product import Product
import numpy as np
import shelve
from gl_utils import draw_gl_polyline,draw_gl_polyline_norm,adjust_gl_view,clear_gl_screen,draw_gl_point,draw_gl_points,draw_gl_point_norm,draw_gl_points_norm,basic_gl_setup,cvmat_to_glmat, redraw_gl_texture
from methods import normalize,denormalize
import atb
import audio
from ctypes import c_int,c_bool,c_float,create_string_buffer

from OpenGL.GL import *
from OpenGL.GLU import gluOrtho2D
import collections

from glfw import *
from plugin import Plugin
import logging
logger = logging.getLogger(__name__)

from square_marker_detect import detect_markers_robust,detect_markers_simple, draw_markers,m_marker_to_screen
from reference_surface import Reference_Surface
from math import sqrt
from timed_counter import TimedCounter

# window calbacks
def on_resize(window,w, h):
  active_window = glfwGetCurrentContext()
  glfwMakeContextCurrent(window)
  adjust_gl_view(w,h)
  glfwMakeContextCurrent(active_window)

class Shopping_Detector(Plugin):
  def __init__(self,g_pool,atb_pos=(0,0), numTrees = 10, numChecks = 50, knnk = 2, trainingProducts = []):
    super(Shopping_Detector, self).__init__()

    print 'network'
    self.context = zmq.Context()
    self.socket = self.context.socket(zmq.PUB)
    self.address = create_string_buffer('tcp://127.0.0.1:5000', 512)
    print 'acquired network connection'
    self.set_server(self.address)

    help_str = "Shopping Message server: Using ZMQ and the *Publish-Subscribe* scheme"
    self.g_pool = g_pool
    self.loadedProducts = False
    self.trained = False
    # mapper init
    print 'sift created'
    self.detector = cv2.SIFT()
    FLANN_INDEX_KDTREE = 6
    FLANN_INDEX_KDTREE = 0
    self.index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = numTrees, table_number = 6, key_size = 12, multi_probe_level = 2)
    self.index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = numTrees)
    self.search_params = dict(checks = numChecks)
    self.matcher = cv2.FlannBasedMatcher(self.index_params, self.search_params)
    
    self.trainingProducts = trainingProducts
    self.create_training_products()
    '''
    self.trainingProducts.append(Product('/Users/mobileexperiencelab/Downloads/digboston.jpg'))
    self.trainingProducts.append(Product('/Users/mobileexperiencelab/Downloads/jobs.png'))
    self.trainingProducts.append(Product('/Users/mobileexperiencelab/Downloads/espn.jpg'))
    self.trainingProducts.append(Product('/Users/mobileexperiencelab/Downloads/a.jpg'))
    '''
    self.generate_training_data()

    self.frameCount = 0
    self.mappedAreas = []
    self.knnk = c_int(knnk)
    self.minGoodMatches = c_int(4) 
    self.maxAvgDistance = c_int(150)
    self.homographyThreshold = c_float(10.0)
    self.distanceWeight = .7
    self.recentGazePoints = collections.deque([], 3)
    self.recent_products_queue = TimedCounter(7.5)

    self.reco_thread = None

    #multi monitor setup
    self.window_should_open = False
    self.window_should_close = False
    self._window = None
    self.fullscreen = c_bool(0)
    self.monitor_idx = c_int(0)
    self.monitor_handles = glfwGetMonitors()
    self.monitor_names = [glfwGetMonitorName(m) for m in self.monitor_handles]
    monitor_enum = atb.enum("Monitor",dict(((key,val) for val,key in enumerate(self.monitor_names))))
    #primary_monitor = glfwGetPrimaryMonitor()

    atb_label = "Product Detection"
    self._bar = atb.Bar(name =self.__class__.__name__, label=atb_label,
      help="product detection parameters", color=(50, 50, 50), alpha=100,
      text='light', position=atb_pos,refresh=.3, size=(300, 100))
    self._bar.add_var("monitor",self.monitor_idx, vtype=monitor_enum,group="Window",)
    self._bar.add_var("fullscreen", self.fullscreen,group="Window")
    self._bar.add_button("  open Window   ", self.do_open, key='m',group="Window")
    self._bar.add_var("min good matches",self.minGoodMatches, step=1,min=0,group="Detector")
    self._bar.add_var("k-nearest-neighbors",self.knnk, step=1,min=1,group="Detector")
    self._bar.add_var("homogoraphy threshold",self.homographyThreshold, step=.1,min=1,group="Detector")
    self._bar.add_var("max distance of good matches",self.maxAvgDistance, step=1,min=1,max=500,group="Detector")
    self._bar.define("valueswidth=170")
    self._bar.add_var("server address",self.address, getter=lambda:self.address, setter=self.set_server)
    self._bar.add_button("close", self.close)

    atb_pos = atb_pos[0],atb_pos[1]+110
    self._bar_markers = atb.Bar(name =self.__class__.__name__+'markers', label='registered surfaces',
      help="list of registered ref surfaces", color=(50, 100, 50), alpha=100,
      text='light', position=atb_pos,refresh=.3, size=(300, 120))

  def set_server(self, new_address):
    try:
      self.socket.bind(new_address.value)
      self.address.value = new_address.value
    except zmq.ZMQError:
      logger.error("Could not set Socket.")

  def close(self):
    self.alive = False

  def cleanup(self):
    """gets called when the plugin get terminated.
       either volunatily or forced.
    """
    self._bar.destroy()
    self.context.destroy()


  def create_training_products(self):
    if not self.loadedProducts:
      self.trainingProducts = []
      productsDir = os.path.dirname(os.path.realpath(__file__)) + '/products'
      productsDir = '/Users/mobileexperiencelab/Downloads/detect/products'
      productsDir = '/Users/mobileexperiencelab/Downloads/Poster1'
      productsDir = '/Users/mobileexperiencelab/Downloads/Poster2'
      for f in os.listdir(productsDir):
        if f.endswith('.jpg'):
          self.trainingProducts.append(Product(productsDir + '/' + f))
    self.loadedProducts = True

  def generate_training_data(self, *products):
    if not self.trained:
      self.trainingData = []
      if len(products) > 0:
        trainingProducts = products
      else:
        trainingProducts= self.trainingProducts
    self.trained = True

    for product in trainingProducts:
      image = cv2.imread(product.imagePath, cv2.CV_LOAD_IMAGE_GRAYSCALE)
      if image is not None and len(image) > 0:
        kp, des = self.detector.detectAndCompute(image, None)
        h, w = image.shape
        corners = np.float32([
            [0, 0],
            [0, h - 1],
            [w - 1, h - 1],
            [w - 1, 0]
        ]).reshape(-1, 1, 2)
        self.trainingData.append({'kp': kp, 'des': des, 'product': product, 'corners': corners})


  def do_open(self):
    if not self._window:
      self.window_should_open = True

  def open_window(self):
    if not self._window:
      if self.fullscreen.value:
        monitor = self.monitor_handles[self.monitor_idx.value]
        mode = glfwGetVideoMode(monitor)
        height,width= mode[0],mode[1]
      else:
        monitor = None
        height,width= 1280,720

      self._window = glfwCreateWindow(height, width, "Reference Surface", monitor=monitor, share=glfwGetCurrentContext())
      if not self.fullscreen.value:
        glfwSetWindowPos(self._window,200,0)

      on_resize(self._window,height,width)

      #Register callbacks
      glfwSetWindowSizeCallback(self._window,on_resize)
      glfwSetKeyCallback(self._window,self.on_key)
      glfwSetWindowCloseCallback(self._window,self.on_close)

      # gl_state settings
      active_window = glfwGetCurrentContext()
      glfwMakeContextCurrent(self._window)
      basic_gl_setup()
      glfwMakeContextCurrent(active_window)

      self.window_should_open = False


  def on_key(self,window, key, scancode, action, mods):
    if not atb.TwEventKeyboardGLFW(key,int(action == GLFW_PRESS)):
      if action == GLFW_PRESS:
        if key == GLFW_KEY_ESCAPE:
          self.on_close()

  def on_close(self,window=None):
    self.window_should_close = True

  def close_window(self):
    if self._window:
      glfwDestroyWindow(self._window)
      self._window = None
      self.window_should_close = False

  def update(self,frame,recent_pupil_positions,events):
    frame = frame.img

    if not self.reco_thread or not self.reco_thread.is_alive():
      self.reco_thread = threading.Thread(target=self.find_areas, args=(frame, self.frameCount, recent_pupil_positions, ))
      self.reco_thread.start()
    '''
    self.find_areas(frame, self.frameCount, recent_pupil_positions)

    for p in recent_pupil_positions:
    if p['norm_pupil'] is not None:
      for s in self.surfaces:
      if s.detected:
        p['realtime gaze on '+s.name] = tuple(s.img_to_ref_surface(np.array(p['norm_gaze'])))
    '''


    if self._window:
      # save a local copy for when we display gaze for debugging on ref surface
      self.recent_pupil_positions = recent_pupil_positions

    if self.window_should_close:
      self.close_window()

    if self.window_should_open:
      self.open_window()
    self.frameCount += 1

  def gl_display(self):
    for outline in self.mappedAreas:
      glLineWidth(4)
      glColor4f(1, 0 , 0, 1)
      glBegin(GL_LINE_LOOP)
      for x,y in [tuple(area[0]) for area in outline.area[0]]:
        glVertex3f(x,y,0.0)
      glEnd()

  def find_areas(self, frame, framecount, recent_pupil_positions):
    t = time.time()
    self.recentGazePoints.extend([p['norm_gaze'] for p in filter(lambda p: 'norm_gaze' in p and not p['norm_gaze'] is None, recent_pupil_positions)])
    (h, w, z) = frame.shape
    minX = w 
    minY = h 
    maxX = 0 
    maxY = 0 
    for p in self.recentGazePoints:
      if not p is None:
        x = (p[0] + .5) * w
        y = (p[1] + .5) * h
        if x < minX: minX = max(0, x - 50)
        if x > maxX: maxX = min(w, x + 50)
        if y < minY: minY = max(0, y - 50)
        if y > maxY: maxY = min(h, y + 50)

    frameBW = cv2.cvtColor(np.asarray(frame[:,:]), cv2.COLOR_BGR2GRAY)
    mask = None
    if len(self.recentGazePoints) > 1:
      mask = np.zeros(frameBW.shape[:2],dtype = np.uint8)
      cv2.fillPoly(mask, [np.array([[minX, minY], [maxX, minY], [maxX, maxY], [minX, maxY]], 'int32').reshape(-1, 1, 2)], 255)
      cv2.imwrite('mask.jpg', mask);
    mask = None

    kp, des = self.detector.detectAndCompute(frameBW, mask)
    mappedAreas = []
    products_seen = 0
    for data in self.trainingData:
      if not des is None:
        matches = self.matcher.knnMatch(data['des'], des, k = self.knnk.value)
        good = []
        for m in matches:
          if len(m) >= 2:
            if m[0].distance < self.distanceWeight * m[1].distance:
              good.append(m[0])
        avg_distance = np.average([ m.distance for m in good ])
        if len(good) > self.minGoodMatches.value and avg_distance < self.maxAvgDistance.value:
          products_seen += 1
          src_pts = np.float32([ data['kp'][m.queryIdx].pt for m in good ]).reshape(-1, 1, 2)
          dst_pts = np.float32([ kp[m.trainIdx].pt for m in good ]).reshape(-1, 1, 2)
          M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.homographyThreshold.value)
          scene_area = [np.int32(cv2.perspectiveTransform(data['corners'].copy(), M))]
          if self.close_to_square(scene_area[0]) and self.gazeInPoly(self.recentGazePoints, scene_area[0], w, h):
            outline = ImageOutline(scene_area, data['product'].outlineColor)
            print cv2.contourArea(scene_area[0])
            mappedAreas.append(outline)
            '''
            if cv2.pointPolygonTest(np.float32(scene_area), last_gaze, False) == 1:
              print data['product'].imagePath
            '''
            path, filename = os.path.split(data['product'].imagePath)
            self.recent_products_queue.add(filename)
            self.check_recent_products()
      if time.time() - t > 3:
        break

    self.mappedAreas = mappedAreas
    print 'found', products_seen, 'products in', time.time() - t, 'of', len(self.trainingData), 'total products'


  def check_recent_products(self):
    probableProduct = self.recent_products_queue.most_common()
    if not probableProduct is None:
      product, count = probableProduct[0]
      print probableProduct
      if count >= 2:
        print '********************************BUY:', product
        self.socket.send(product)
        audio.say('purchased' + product)
        self.recent_products_queue.clear()
        return

  def gazeInPoly(self, gazes, points, w, h):
    for gaze in gazes:
      if not gaze is None:
        x = (gaze[0]) * w
        y = (gaze[1]) * h
        print x, y
        lessThanX = 0
        lessThanY = 0
        for point in points:
          point = point[0]
          if x >= point[0]: lessThanX += 1
          if y >= point[1]: lessThanY += 1
        if lessThanX == 2 and lessThanY == 2:
          return True
    return False
    

  
  def close_to_square(self, points):
    for point in points:
      point = point[0]
    return True


  def cleanup(self):
    """ called when the plugin gets terminated.
    This happends either voluntary or forced.
    if you have an atb bar or glfw window destroy it here.
    """

    if self._window:
      self.close_window()
    self._bar.destroy()
    self._bar_markers.destroy()
    self.context.destroy()

