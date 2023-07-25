import json
import os
import random

import select
import threading
import time
import socket
import fcntl
import struct
from threading import Thread
from cereal import messaging, log
from common.numpy_fast import clip
from common.realtime import sec_since_boot
from common.conversions import Conversions as CV

CAMERA_SPEED_FACTOR = 0.99


class Port:
  BROADCAST_PORT = 2899
  RECEIVE_PORT = 2843
  LOCATION_PORT = BROADCAST_PORT


class RoadLimitSpeedServer:
  def __init__(self):
    self.json_road_limit = None
    self.json_apilot = None
    self.active = 0
    self.active_apilot = 0
    self.last_updated = 0
    self.last_updated_apilot = 0
    self.last_updated_active = 0
    self.last_exception = None
    self.lock = threading.Lock()
    self.remote_addr = None

    self.remote_gps_addr = None
    self.last_time_location = 0

    broadcast = Thread(target=self.broadcast_thread, args=[])
    broadcast.setDaemon(True)
    broadcast.start()

    self.gps_sm = messaging.SubMaster(['gpsLocationExternal'], poll=['gpsLocationExternal'])
    self.gps_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    self.gps_event = threading.Event()
    gps_thread = Thread(target=self.gps_thread, args=[])
    gps_thread.setDaemon(True)
    gps_thread.start()

  def gps_thread(self):
    try:
      period = 1.0
      wait_time = period
      i = 0.
      frame = 1
      start_time = sec_since_boot()
      while True:
        self.gps_event.wait(wait_time)
        self.gps_timer()

        now = sec_since_boot()
        error = (frame * period - (now - start_time))
        i += error * 0.1
        wait_time = period + error * 0.5 + i
        wait_time = clip(wait_time, 0.8, 1.0)
        frame += 1

    except:
      pass

  def gps_timer(self):
    try:
      if self.remote_gps_addr is not None:
        self.gps_sm.update(0)
        if self.gps_sm.updated['gpsLocationExternal']:
          location = self.gps_sm['gpsLocationExternal']

          if location.accuracy < 10.:
            json_location = json.dumps({"location": [
              location.latitude,
              location.longitude,
              location.altitude,
              location.speed,
              location.bearingDeg,
              location.accuracy,
              location.timestamp,
              # location.source,
              # location.vNED,
              location.verticalAccuracy,
              location.bearingAccuracyDeg,
              location.speedAccuracy,
            ]})

            address = (self.remote_gps_addr[0], Port.LOCATION_PORT)
            self.gps_socket.sendto(json_location.encode(), address)
    except:
      self.remote_gps_addr = None

  def get_broadcast_address(self):
    try:
      s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
      ip = fcntl.ioctl(
        s.fileno(),
        0x8919,
        struct.pack('256s', 'wlan0'.encode('utf-8'))
      )[20:24]

      return socket.inet_ntoa(ip)
    except:
      return None

  def broadcast_thread(self):

    broadcast_address = None
    frame = 0

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
      try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        while True:

          try:

            if broadcast_address is None or frame % 10 == 0:
              broadcast_address = self.get_broadcast_address()

            #print('broadcast_address', broadcast_address)

            if broadcast_address is not None:
              address = (broadcast_address, Port.BROADCAST_PORT)
              sock.sendto('EON:ROAD_LIMIT_SERVICE:v1'.encode(), address)
          except:
            pass

          time.sleep(5.)
          frame += 1

      except:
        pass

  def send_sdp(self, sock):
    try:
      sock.sendto('EON:ROAD_LIMIT_SERVICE:v1'.encode(), (self.remote_addr[0], Port.BROADCAST_PORT))
      #print(self.remote_addr[0])
      sock.sendto('EON:ROAD_LIMIT_SERVICE:v1'.encode(), (self.remote_addr[0], 2898))
    except:
      pass

  def udp_recv(self, sock):
    ret = False
    try:
      ready = select.select([sock], [], [], 0.5)
      ret = bool(ready[0])
      if ret:
        data, self.remote_addr = sock.recvfrom(2048)
        json_obj = json.loads(data.decode())
        print(json_obj)

        if 'cmd' in json_obj:
          try:
            os.system(json_obj['cmd'])
            ret = False
          except:
            pass

        if 'request_gps' in json_obj:
          try:
            if json_obj['request_gps'] == 1:
              self.remote_gps_addr = self.remote_addr
            else:
              self.remote_gps_addr = None
            ret = False
          except:
            pass

        if 'echo' in json_obj:
          try:
            echo = json.dumps(json_obj["echo"])
            sock.sendto(echo.encode(), (self.remote_addr[0], Port.BROADCAST_PORT))
            ret = False
          except:
            pass

        try:
          self.lock.acquire()
          try:
            if 'active' in json_obj:
              self.active = json_obj['active']
              self.last_updated_active = sec_since_boot()
          except:
            pass

          if 'road_limit' in json_obj:
            self.json_road_limit = json_obj['road_limit']
            self.last_updated = sec_since_boot()

          if 'apilot' in json_obj:
            self.json_apilot = json_obj['apilot']
            self.last_updated_apilot = sec_since_boot()

        finally:
          self.lock.release()

    except:

      try:
        self.lock.acquire()
        self.json_road_limit = None
      finally:
        self.lock.release()

    return ret

  def check(self):
    now = sec_since_boot()
    if now - self.last_updated > 6.:
      try:
        self.lock.acquire()
        self.json_road_limit = None
      finally:
        self.lock.release()

    if now - self.last_updated_apilot > 6.:
      try:
        self.lock.acquire()
        self.json_apilot = None
      finally:
        self.lock.release()

    if now - self.last_updated_active > 6.:
      self.active = 0
    if now - self.last_updated_apilot > 6.:
      self.active_apilot = 0


  def get_limit_val(self, key, default=None):
    return self.get_json_val(self.json_road_limit, key, default)

  def get_apilot_val(self, key, default=None):
    return self.get_json_val(self.json_apilot, key, default)


  def get_json_val(self, json, key, default=None):

    try:
      if json is None:
        return default

      if key in json:
        return json[key]

    except:
      pass

    return default


def main():
  server = RoadLimitSpeedServer()
  roadLimitSpeed = messaging.pub_sock('roadLimitSpeed')

  sock_carState = messaging.sub_sock("carState")
  carState = None

  xTurnInfo = -1
  xDistToTurn = -1
  xSpdDist = -1
  xSpdLimit = -1
  xSignType = -1
  xRoadSignType = -1
  xRoadLimitSpeed = -1
  xRoadName = ""

  xBumpDistance = 0
  xTurnInfo_prev = xTurnInfo
  sdiDebugText = ""

  sdi_valid_count = 0
  apn_valid_count = 0
  sdiType = -1

  totalDistance = 0.0

  with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
    try:
      try:
        sock.bind(('0.0.0.0', 843))
      except:
        sock.bind(('0.0.0.0', Port.RECEIVE_PORT))

      sock.setblocking(False)

      while True:

        server.udp_recv(sock)

        try:
          dat = messaging.recv_sock(sock_carState, wait=False)
          if dat is not None:
            carState = dat.carState
        except:
          pass

        dat = messaging.new_message()
        dat.init('roadLimitSpeed')
        dat.roadLimitSpeed.active = server.active
        dat.roadLimitSpeed.roadLimitSpeed = server.get_limit_val("road_limit_speed", 0)
        dat.roadLimitSpeed.isHighway = server.get_limit_val("is_highway", False)
        dat.roadLimitSpeed.camType = server.get_limit_val("cam_type", 0)
        dat.roadLimitSpeed.camLimitSpeedLeftDist = server.get_limit_val("cam_limit_speed_left_dist", 0)
        dat.roadLimitSpeed.camLimitSpeed = server.get_limit_val("cam_limit_speed", 0)
        dat.roadLimitSpeed.sectionLimitSpeed = server.get_limit_val("section_limit_speed", 0)
        dat.roadLimitSpeed.sectionLeftDist = server.get_limit_val("section_left_dist", 0)
        dat.roadLimitSpeed.sectionAvgSpeed = server.get_limit_val("section_avg_speed", 0)
        dat.roadLimitSpeed.sectionLeftTime = server.get_limit_val("section_left_time", 0)
        dat.roadLimitSpeed.sectionAdjustSpeed = server.get_limit_val("section_adjust_speed", False)
        dat.roadLimitSpeed.camSpeedFactor = server.get_limit_val("cam_speed_factor", CAMERA_SPEED_FACTOR)

        atype = server.get_apilot_val("type")
        value = server.get_apilot_val("value")
        atype = "none" if atype is None else atype
        value = "-1" if value is None else value
        try:
          value_int = int(value)
        except:
          value_int = -100

        now = sec_since_boot()
        #print(atype, value)
        delta_dist = 0.0
        if carState is not None:
          CS = carState
          delta_dist = CS.totalDistance - totalDistance
          totalDistance = CS.totalDistance
          if CS.gasPressed:
            xBumpDistance = -1
            if xSignType == 124:
              xSignType = -1
        apn_valid = True
        if atype == 'none':
          apn_valid = False
        elif atype == 'opkrturninfo':
          xTurnInfo = value_int
        elif atype == 'opkrdistancetoturn':
          xDistToTurn = value_int
        elif atype == 'opkrspddist':
          xSpdDist = value_int
        elif atype == 'opkr-spddist':
          pass
        elif atype == 'opkrspdlimit':
          xSpdLimit = value_int
        elif atype == 'opkr-spdlimit':
          pass
        elif atype == 'opkrsigntype':
          xSignType = value_int
        elif atype == 'opkr-signtype':
          pass
        elif atype == 'opkrroadsigntype':
          xRoadSignType = value_int
        elif atype == 'opkrroadlimitspeed':
          xRoadLimitSpeed = value_int
        elif atype == 'opkrwazecurrentspd':
          pass
        elif atype == 'opkrwazeroadname':
          xRoadName = value
        elif atype == 'opkrwazenavsign':
          if value == '2131230983': # 목적지
            xTurnInfo = -1
          elif value == '2131230988': # turnLeft
            xTurnInfo = 1
          elif value == '2131230989': # turnRight
            xTurnInfo = 2
          elif value == '2131230985': 
            xTurnInfo = 4
          else:
            xTurnInfo = value_int
          xTurnInfo_prev = xTurnInfo
        elif atype == 'opkrwazenavdist':
          xDistToTurn = value_int
          if xTurnInfo<0:
            xTurnInfo = xTurnInfo_prev
        elif atype == 'opkrwazeroadspdlimit':
          xRoadLimitSpeed = value_int
        elif atype == 'opkrwazealertdist':
          pass
        elif atype == 'opkrwazereportid':
          pass
        elif atype == 'apilotman':
          server.active_apilot = 1
        else:
          print("unknown{}={}".format(atype, value))
        #dat.roadLimitSpeed.xRoadName = apilot_val['opkrroadname']['value']

        #for 띠맵
        nTBTTurnType = int(server.get_apilot_val("nTBTTurnType", -1))
        if nTBTTurnType in [12, 16]:
          xTurnInfo = 1  # turn left
        elif nTBTTurnType in [13, 19]:
          xTurnInfo = 2  # turn right
        elif nTBTTurnType in [7, 44, 17, 75, 102, 105, 112, 115, 76, 118]: # left lanechange
          xTurnInfo = 3  # slight left
        elif nTBTTurnType in [6, 43, 73, 74, 101, 104, 111, 114, 123, 124, 117]: # right lanechange
          xTurnInfo = 4  # slight right
        elif nTBTTurnType >= 0:
          xTurnInfo = -1
        nTBTDist = int(server.get_apilot_val("nTBTDist", 0))
        if nTBTDist > 0:
          xDistToTurn = nTBTDist
        nRoadLimitSpeed = int(server.get_apilot_val("nRoadLimitSpeed", -1))
        sdi_valid = True if nRoadLimitSpeed >= 0 else False
        if nRoadLimitSpeed > 0:
          xRoadLimitSpeed = nRoadLimitSpeed
        nSdiType = int(server.get_apilot_val("nSdiType", -1))
        nSdiDist = int(server.get_apilot_val("nSdiDist", -1))
        nSdiSpeedLimit = int(server.get_apilot_val("nSdiSpeedLimit", -1))
        nSdiPlusType = int(server.get_apilot_val("nSdiPlusType", -1))
        nSdiPlusDist = int(server.get_apilot_val("nSdiPlusDist", -1))
        nSdiPlusSpeedLimit = int(server.get_apilot_val("nSdiPlusSpeedLimit", -1))
        #sdiType: 
        # 0: speedLimit, 1: speedLimitPos, 2:SpeedBlockStartPos, 3: SpeedBlockEndPos, 4:SpeedBlockMidPos, 
        # 5: Tail, 6: SignalAccidentPos, 7: SpeedLimitDangerous, 8:BoxSpeedLimit, 9: BusLane, 
        # 10:ChangerRoadPos, 11:RoadControlPos, 12: IntruderArea, 13: TrafficInfoCollectPos, 14:CctvArea
        # 15:OverloadDangerousArea, 16:LoadBadControlPos, 17:ParkingControlPos, 18:OnewayArea, 19:RailwayCrossing
        # 20:SchoolZoneStart, 21:SchoolZoneEnd, 22:SpeedBump, 23:LpgStation, 24:TunnelArea, 
        # 25:ServiceArea
        # 66:ChangableSpeedBlockStartPos, 67:ChangableSpeedBlockEndPos
        if nSdiType in [0,1,2,3,8] and nSdiSpeedLimit > 0: # SpeedLimitPos, nSdiSection: 2,
          xSpdLimit = nSdiSpeedLimit
          xSpdDist = nSdiDist
          sdiType = nSdiType
        elif nSdiPlusType == 22 or nSdiType == 22: # SpeedBump
          xSpdLimit = 35
          xSpdDist = nSdiPlusDist if nSdiPlusType == 22 else nSdiDist
          sdiType = 22
        elif nTBTTurnType >= 0 and nSdiType <= 0 and nSdiPlusType <= 0: # 데이터는 수신되었으나, sdi 수신이 없으면, 감속중 다른곳으로 빠진경우... 초기화...
          xSpedLimit = xSpdDist = sdiType = -1

        if sdiType >= 0:
          dat.roadLimitSpeed.camType = sdiType

        szPosRoadName = server.get_apilot_val("szPosRoadName", "")
        if len(szPosRoadName) > 0:
          xRoadName = szPosRoadName

        sdi_valid_count -= 1
        if sdi_valid:
          sdi_valid_count = 10
          sdiDebugText = "({}/{}/{} {}/{}/{})".format(nSdiType, nSdiDist, nSdiSpeedLimit, nSdiPlusType, nSdiPlusDist, nSdiPlusSpeedLimit)
        apn_valid_count -= 1
        if apn_valid:
          apn_valid_count = 10

        if xTurnInfo >= 0:
          xDistToTurn -= delta_dist
          if xDistToTurn < 0:
            xTurnInfo = -1

        if xSpdLimit >= 0:
          xSpdDist -= delta_dist
          if xSpdDist < 0:
            xSpdLimit = -1

        if xBumpDistance > 0:
          xBumpDistance -= delta_dist
          if xBumpDistance <= 0 and xSignType == 124:
            xSignType = -1
          else:
            dat.roadLimitSpeed.camType = 22 # bump

        if xSignType == 124: ##사고방지턱
          if xBumpDistance <= 0:
            xBumpDistance = 110
        else:
          xBumpDistance = -1

        if sdi_valid_count > 0:
          dat.roadLimitSpeed.active = 200 + server.active
        elif apn_valid_count > 0:
          dat.roadLimitSpeed.active = 100 + server.active
        else:
          xSpdDist = xBumpDistance = xSpdLimit = -1
        #print(dat.roadLimitSpeed.active)
        #print("turn={},{}".format(xTurnInfo, xDistToTurn))
        dat.roadLimitSpeed.xTurnInfo = int(xTurnInfo)
        dat.roadLimitSpeed.xDistToTurn = int(xDistToTurn)
        dat.roadLimitSpeed.xSpdDist = int(xSpdDist) if xBumpDistance <= 0 else int(xBumpDistance)
        dat.roadLimitSpeed.xSpdLimit = int(xSpdLimit) if xBumpDistance <= 0 else 10 # 속도는 추후조절해야함. 벌트는 일단 10
        dat.roadLimitSpeed.xSignType = int(xSignType) if xBumpDistance <= 0 else 22
        dat.roadLimitSpeed.xRoadSignType = int(xRoadSignType)
        dat.roadLimitSpeed.xRoadLimitSpeed = int(xRoadLimitSpeed)
        dat.roadLimitSpeed.xRoadName = xRoadName + sdiDebugText

        roadLimitSpeed.send(dat.to_bytes())
        server.send_sdp(sock)
        server.check()
        time.sleep(0.03)

    except Exception as e:
      print(e)
      server.last_exception = e


class RoadSpeedLimiter:
  def __init__(self):
    self.slowing_down = False
    self.started_dist = 0
    self.session_limit = False

    self.sock = messaging.sub_sock("roadLimitSpeed")
    self.roadLimitSpeed = None

  def recv(self):
    try:
      dat = messaging.recv_sock(self.sock, wait=False)
      if dat is not None:
        self.roadLimitSpeed = dat.roadLimitSpeed
    except:
      pass

  def get_active(self):
    self.recv()
    if self.roadLimitSpeed is not None:
      return self.roadLimitSpeed.active % 100
    return 0

  def get_max_speed(self, cluster_speed, is_metric, autoNaviSpeedCtrlStart=22, autoNaviSpeedCtrlEnd=6, autoNaviSpeedBumpDist=10):

    log = ""
    self.recv()

    if self.roadLimitSpeed is None:
      return 0, 0, 0, False, ""

    try:
      road_limit_speed = self.roadLimitSpeed.roadLimitSpeed
      is_highway = self.roadLimitSpeed.isHighway

      cam_type = int(self.roadLimitSpeed.camType)

      cam_limit_speed_left_dist = self.roadLimitSpeed.camLimitSpeedLeftDist
      cam_limit_speed = self.roadLimitSpeed.camLimitSpeed

      if self.roadLimitSpeed.xSpdLimit > 0 and self.roadLimitSpeed.xSpdDist > 0:
        cam_limit_speed_left_dist = self.roadLimitSpeed.xSpdDist
        cam_limit_speed = self.roadLimitSpeed.xSpdLimit
        self.session_limit = True if (self.roadLimitSpeed.xSignType == 165) or (cam_limit_speed_left_dist > 3000) else False
        #log = "limit={:.1f},{:.1f}".format(self.roadLimitSpeed.xSpdLimit, self.roadLimitSpeed.xSpdDist)

        self.session_limit = False if cam_limit_speed_left_dist < 50 else self.session_limit

      section_limit_speed = self.roadLimitSpeed.sectionLimitSpeed
      section_left_dist = self.roadLimitSpeed.sectionLeftDist
      section_avg_speed = self.roadLimitSpeed.sectionAvgSpeed
      section_left_time = self.roadLimitSpeed.sectionLeftTime
      section_adjust_speed = self.roadLimitSpeed.sectionAdjustSpeed

      camSpeedFactor = clip(self.roadLimitSpeed.camSpeedFactor, 0.98, 1.0)
      if False and is_highway is not None:
        if is_highway:
          MIN_LIMIT = 40
          MAX_LIMIT = 120
        else:
          MIN_LIMIT = 20
          MAX_LIMIT = 100
      else:
        MIN_LIMIT = 20
        MAX_LIMIT = 120

      if cam_type == 22:  # speed bump
        MIN_LIMIT = 10

      if cam_limit_speed_left_dist is not None and cam_limit_speed is not None and cam_limit_speed_left_dist > 0:

        v_ego = cluster_speed * (CV.KPH_TO_MS if is_metric else CV.MPH_TO_MS)
        diff_speed = cluster_speed - (cam_limit_speed * camSpeedFactor)
        #cam_limit_speed_ms = cam_limit_speed * (CV.KPH_TO_MS if is_metric else CV.MPH_TO_MS)

        #starting_dist = v_ego * 30.
        starting_dist = v_ego * autoNaviSpeedCtrlStart

        if cam_type == 22:
          starting_dist = v_ego * 6
          safe_dist = autoNaviSpeedBumpDist #v_ego * 0.5 # speed bump
        else:
          safe_dist = v_ego * autoNaviSpeedCtrlEnd

        log = "SPDCTRL({})={:.0f}<{:.0f}<{:.0f},type={},{:.0f}".format(self.slowing_down, safe_dist, cam_limit_speed_left_dist, starting_dist, cam_type, self.started_dist)

        if MIN_LIMIT <= cam_limit_speed <= MAX_LIMIT and (self.slowing_down or cam_limit_speed_left_dist < starting_dist):
          if not self.slowing_down:
            self.started_dist = cam_limit_speed_left_dist
            self.slowing_down = True
            first_started = True
          else:
            first_started = False

          td = self.started_dist - safe_dist
          d = cam_limit_speed_left_dist - safe_dist

          if d > 0. and td > 0. and diff_speed > 0. and (section_left_dist is None or section_left_dist < 10 or cam_type == 2) and not self.session_limit:
            pp = (d / td) ** 0.6
          else:
            pp = 0

          return cam_limit_speed * camSpeedFactor + int(pp * diff_speed), \
                 cam_limit_speed, cam_limit_speed_left_dist, first_started, log

        self.slowing_down = False
        return 0, cam_limit_speed, cam_limit_speed_left_dist, False, log

      elif section_left_dist is not None and section_limit_speed is not None and section_left_dist > 0:
        if MIN_LIMIT <= section_limit_speed <= MAX_LIMIT:

          if not self.slowing_down:
            self.slowing_down = True
            first_started = True
          else:
            first_started = False

          speed_diff = 0
          if section_adjust_speed is not None and section_adjust_speed:
            speed_diff = (section_limit_speed - section_avg_speed) / 2.

          return section_limit_speed * camSpeedFactor + speed_diff, section_limit_speed, section_left_dist, first_started, log

        self.slowing_down = False
        return 0, section_limit_speed, section_left_dist, False, log

    except Exception as e:
      log = "Ex: " + str(e)
      pass

    self.slowing_down = False
    return 0, 0, 0, False, log


road_speed_limiter = None


def road_speed_limiter_get_active():
  global road_speed_limiter
  if road_speed_limiter is None:
    road_speed_limiter = RoadSpeedLimiter()

  return road_speed_limiter.get_active()


def road_speed_limiter_get_max_speed(cluster_speed, is_metric):
  global road_speed_limiter
  if road_speed_limiter is None:
    road_speed_limiter = RoadSpeedLimiter()

  return road_speed_limiter.get_max_speed(cluster_speed, is_metric)


def get_road_speed_limiter():
  global road_speed_limiter
  if road_speed_limiter is None:
    road_speed_limiter = RoadSpeedLimiter()
  return road_speed_limiter


if __name__ == "__main__":
  main()
