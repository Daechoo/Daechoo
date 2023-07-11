import numpy as np
from cereal import log, car
from common.conversions import Conversions as CV
from common.realtime import DT_MDL
from common.params import Params

LaneChangeState = log.LateralPlan.LaneChangeState
LaneChangeDirection = log.LateralPlan.LaneChangeDirection
EventName = car.CarEvent.EventName

LANE_CHANGE_SPEED_MIN = 20 * CV.MPH_TO_MS
LANE_CHANGE_TIME_MAX = 10.

DESIRES = {
  LaneChangeDirection.none: {
    LaneChangeState.off: log.LateralPlan.Desire.none,
    LaneChangeState.preLaneChange: log.LateralPlan.Desire.none,
    LaneChangeState.laneChangeStarting: log.LateralPlan.Desire.none,
    LaneChangeState.laneChangeFinishing: log.LateralPlan.Desire.none,
  },
  LaneChangeDirection.left: {
    LaneChangeState.off: log.LateralPlan.Desire.none,
    LaneChangeState.preLaneChange: log.LateralPlan.Desire.none,
    LaneChangeState.laneChangeStarting: log.LateralPlan.Desire.laneChangeLeft,
    LaneChangeState.laneChangeFinishing: log.LateralPlan.Desire.laneChangeLeft,
  },
  LaneChangeDirection.right: {
    LaneChangeState.off: log.LateralPlan.Desire.none,
    LaneChangeState.preLaneChange: log.LateralPlan.Desire.none,
    LaneChangeState.laneChangeStarting: log.LateralPlan.Desire.laneChangeRight,
    LaneChangeState.laneChangeFinishing: log.LateralPlan.Desire.laneChangeRight,
  },
}


class DesireHelper:
  def __init__(self):
    self.lane_change_state = LaneChangeState.off
    self.lane_change_direction = LaneChangeDirection.none
    self.lane_change_timer = 0.0
    self.lane_change_ll_prob = 1.0
    self.keep_pulse_timer = 0.0
    self.prev_one_blinker = False
    self.desire = log.LateralPlan.Desire.none

    self.paramsCount = 100
    self.update_params()
    self.prev_leftBlinker = False
    self.prev_rightBlinker = False
    self.desireEvent = 0
    self.needTorque = False

  def update_(self, carstate, lateral_active, lane_change_prob):
    v_ego = carstate.vEgo
    one_blinker = carstate.leftBlinker != carstate.rightBlinker
    below_lane_change_speed = v_ego < LANE_CHANGE_SPEED_MIN

    if not lateral_active or self.lane_change_timer > LANE_CHANGE_TIME_MAX:
      self.lane_change_state = LaneChangeState.off
      self.lane_change_direction = LaneChangeDirection.none
    else:
      # LaneChangeState.off
      if self.lane_change_state == LaneChangeState.off and one_blinker and not self.prev_one_blinker and not below_lane_change_speed:
        self.lane_change_state = LaneChangeState.preLaneChange
        self.lane_change_ll_prob = 1.0

      # LaneChangeState.preLaneChange
      elif self.lane_change_state == LaneChangeState.preLaneChange:
        # Set lane change direction
        self.lane_change_direction = LaneChangeDirection.left if \
          carstate.leftBlinker else LaneChangeDirection.right

        torque_applied = carstate.steeringPressed and \
                         ((carstate.steeringTorque > 0 and self.lane_change_direction == LaneChangeDirection.left) or
                          (carstate.steeringTorque < 0 and self.lane_change_direction == LaneChangeDirection.right))

        blindspot_detected = ((carstate.leftBlindspot and self.lane_change_direction == LaneChangeDirection.left) or
                              (carstate.rightBlindspot and self.lane_change_direction == LaneChangeDirection.right))

        if not one_blinker or below_lane_change_speed:
          self.lane_change_state = LaneChangeState.off
          self.lane_change_direction = LaneChangeDirection.none
        elif torque_applied and not blindspot_detected:
          self.lane_change_state = LaneChangeState.laneChangeStarting

      # LaneChangeState.laneChangeStarting
      elif self.lane_change_state == LaneChangeState.laneChangeStarting:
        # fade out over .5s
        self.lane_change_ll_prob = max(self.lane_change_ll_prob - 2 * DT_MDL, 0.0)

        # 98% certainty
        if lane_change_prob < 0.02 and self.lane_change_ll_prob < 0.01:
          self.lane_change_state = LaneChangeState.laneChangeFinishing

      # LaneChangeState.laneChangeFinishing
      elif self.lane_change_state == LaneChangeState.laneChangeFinishing:
        # fade in laneline over 1s
        self.lane_change_ll_prob = min(self.lane_change_ll_prob + DT_MDL, 1.0)

        if self.lane_change_ll_prob > 0.99:
          self.lane_change_direction = LaneChangeDirection.none
          if one_blinker:
            self.lane_change_state = LaneChangeState.preLaneChange
          else:
            self.lane_change_state = LaneChangeState.off

    if self.lane_change_state in (LaneChangeState.off, LaneChangeState.preLaneChange):
      self.lane_change_timer = 0.0
    else:
      self.lane_change_timer += DT_MDL

    self.prev_one_blinker = one_blinker

    self.desire = DESIRES[self.lane_change_direction][self.lane_change_state]

    # Send keep pulse once per second during LaneChangeStart.preLaneChange
    if self.lane_change_state in (LaneChangeState.off, LaneChangeState.laneChangeStarting):
      self.keep_pulse_timer = 0.0
    elif self.lane_change_state == LaneChangeState.preLaneChange:
      self.keep_pulse_timer += DT_MDL
      if self.keep_pulse_timer > 1.0:
        self.keep_pulse_timer = 0.0
      elif self.desire in (log.LateralPlan.Desire.keepLeft, log.LateralPlan.Desire.keepRight):
        self.desire = log.LateralPlan.Desire.none


  def update_params(self):
    self.paramsCount += 1
    if self.paramsCount > 100:
      self.autoTurnControl = int(Params().get("AutoTurnControl", encoding="utf8"))
      self.autoLaneChangeSpeed = int(Params().get("AutoLaneChangeSpeed", encoding="'utf8"))
      self.paramsCount = 0

  def detect_road_edge(self, md):
    left_edge_prob = np.clip(1.0 - md.roadEdgeStds[0], 0.0, 1.0)
    left_nearside_prob = md.laneLineProbs[0]
    left_close_prob = md.laneLineProbs[1]
    right_close_prob = md.laneLineProbs[2]
    right_nearside_prob = md.laneLineProbs[3]
    right_edge_prob = np.clip(1.0 - md.roadEdgeStds[1], 0.0, 1.0)

    if right_edge_prob > 0.35 and right_nearside_prob < 0.2 and left_nearside_prob >= right_nearside_prob:
      return 1 # right roadedge detected
    elif left_edge_prob > 0.35 and left_nearside_prob < 0.2 and right_nearside_prob >= left_nearside_prob:
      return -1 #left roadedge detected
    return 0

  def nav_update(self, carstate, roadLimitSpeed, road_edge_stat):
    direction = nav_direction = LaneChangeDirection.none
    nav_turn = False
    nav_event = 0
    need_torque = False
    if self.autoTurnControl > 0:
      nav_distance = roadLimitSpeed.xDistToTurn
      nav_type = roadLimitSpeed.xTurnInfo
      nav_turn = True if nav_type in [1,2] else False
      direction = LaneChangeDirection.left if nav_type in [1,3] else LaneChangeDirection.right if nav_type in [2,4,43] else LaneChangeDirection.none
  
    nav_direction = LaneChangeDirection.none
    if 5 < nav_distance < 200:
      if self.desireReady >= 0: # -1이면 현재의 네비정보는 사용안함.
        self.desireReady = 1
        if nav_turn:
          if nav_distance < 60: # 턴시작
            nav_direction = direction
          # 로드에지가 검출안되면 차로변경(토크필요), 그외 차로변경 명령
          else: # nav_distance < 180 and (direction == LaneChangeDirection.right) and (road_edge_stat < 1) and not carstate.rightBlindspot and self.navActive==0: # 멀리있는경우 차로변경
            need_torque = True
            nav_turn = False
            nav_direction = direction
        elif nav_distance < 180: # and self.navActive == 0: # 차로변경시작
          need_torque = True
          nav_direction = direction
        nav_event = EventName.audioTurn if nav_turn else EventName.audioLaneChange
      else:
        nav_turn = False
    else:
      nav_turn = False
      self.desireReady = 0
      direction = LaneChangeDirection.none

    return nav_direction, nav_turn, need_torque, nav_event

  def update(self, carstate, lateral_active, lane_change_prob, md, turn_prob, roadLimitSpeed, lane_width):
    self.update_params()
    v_ego = carstate.vEgo
    v_ego_kph = v_ego * CV.MS_TO_KPH

    road_edge_stat = self.detect_road_edge(md)
    if self.autoTurnControl > 0:
      nav_direction, nav_turn, need_torque, nav_event = self.nav_update(carstate, roadLimitSpeed, road_edge_stat)
    else:
      nav_direction = LaneChangeDirection.none
      nav_turn = False
      nav_event = 0
      need_torque = False

    leftBlinker = carstate.leftBlinker
    rightBlinker = carstate.rightBlinker
    trig_leftBlinker = True if carstate.leftBlinker and not self.prev_leftBlinker else False
    trig_rightBlinker = True if carstate.rightBlinker and not self.prev_rightBlinker else False

    if nav_direction == LaneChangeDirection.right:
      if leftBlinker:
        nav_direction = LaneChangeDirection.none
        self.desireReady = -1
      else:
        rightBlinker = True
    elif nav_direction == LaneChangeDirection.left:
      if rightBlinker:
        nav_direction = LaneChangeDirection.none
        self.desireReady = -1
      else:
        leftBlinker = True

    ## nav것과 carstate것과 같이 사용함.
    one_blinker = leftBlinker != rightBlinker
    ## 네비가 켜지면 강제로 상태변경
    if one_blinker and nav_direction != LaneChangeDirection.none:
      self.prev_one_blinker = False
      
    ## 핸들토크가 조향방향으로 가해짐.
    torque_applied = carstate.steeringPressed and \
                        ((carstate.steeringTorque > 0 and leftBlinker) or
                        (carstate.steeringTorque < 0 and rightBlinker))

    ## 핸들토크가 반대방향으로 가해짐.
    steering_pressed = carstate.steeringPressed and \
                        ((carstate.steeringTorque > 0 and leftBlinker) or
                        (carstate.steeringTorque < 0 and rightBlinker))

    blindspot_detected = ((carstate.leftBlindspot and leftBlinker) or
                          (carstate.rightBlindspot and rightBlinker))

    roadedge_detected = ((road_edge_stat == -1 and leftBlinker) or
                          (road_edge_stat == 1 and rightBlinker))

    self.desire = log.LateralPlan.Desire.none
    self.desireEvent = 0
    if not lateral_active or self.lane_change_timer > LANE_CHANGE_TIME_MAX:
      self.lane_change_state = LaneChangeState.off
      self.lane_change_direction = LaneChangeDirection.none
    else:
      if self.lane_change_state == LaneChangeState.off and one_blinker and not self.prev_one_blinker: # and not below_lane_change_speed:
        self.lane_change_state = LaneChangeState.preLaneChange
        self.lane_change_ll_prob = 1.0

      elif self.lane_change_state == LaneChangeState.preLaneChange:
        self.lane_change_direction = LaneChangeDirection.left if leftBlinker else LaneChangeDirection.right
        if not one_blinker:
          self.lane_change_state == LaneChangeState.off
          self.lane_change_direction = LaneChangeDirection.none
        else:
          if nav_turn:
            self.lane_change_state = LaneChangeState.laneChangeStarting
          else:
            if nav_direction == LaneChangeDirection.none: # 수동깜박이
              if blindspot_detected:
                self.desireEvent = EventName.laneChangeBlocked
              elif roadedge_detected:
                self.desireEvent = EventName.laneChangeRoadEdge
              else:
                if leftBlinker and v_ego_kph < self.autoLaneChangeSpeed: # 저속에 좌측차로변경 토크필요
                  need_torque = True
                self.lane_change_state = LaneChangeState.laneChangeStarting
            else: # 네비..
              if blindspot_detected or roadedge_detected:
                if need_torque or self.needTorque: # 이벤트때문에.... 
                  self.lane_change_state = LaneChangeState.laneChangeStarting
                pass
              else:
                if nav_direction == LaneChangeDirection.right and trig_rightBlinker: # 대기중 우측깜박이를 켜면, 토크검사생략하고 작동시작.
                  need_torque = False
                self.lane_change_state = LaneChangeState.laneChangeStarting

          ## 핸들토크검사
          if self.lane_change_state == LaneChangeState.laneChangeStarting:
            if need_torque or self.needTorque:
              if torque_applied:
                self.needTorque = False
              else:
                self.lane_change_state = LaneChangeState.preLaneChange
                if self.desireEvent == 0:
                  self.desireEvent = EventName.preLaneChangeLeft if self.lane_change_direction == LaneChangeDirection.left else EventName.preLaneChangeRight
          else:
            self.needTorque = False


          #if nav_event > 0 and self.desireEvent == 0:
          #  self.desireEvent = nav_event

      # LaneChangeState.laneChangeStarting
      elif self.lane_change_state == LaneChangeState.laneChangeStarting:
        # fade out over .5s
        self.lane_change_ll_prob = max(self.lane_change_ll_prob - 2 * DT_MDL, 0.0) ## *2씩 뺐으니, 0.5초,

        if nav_turn:
          self.desire = log.LateralPlan.Desire.turnLeft if self.lane_change_direction == LaneChangeDirection.left else log.LateralPlan.Desire.turnRight
        else:
          self.desire = log.LateralPlan.Desire.laneChangeLeft if self.lane_change_direction == LaneChangeDirection.left else log.LateralPlan.Desire.laneChangeRight

        # 98% certainty
        if lane_change_prob < 0.02 and self.lane_change_ll_prob < 0.01: # 0.5초가 지난후부터 차선변경이 완료되었는지확인.
          self.lane_change_state = LaneChangeState.laneChangeFinishing

        if steering_pressed:
          self.lane_change_state = LaneChangeState.off
          self.desireReady = -1
      # LaneChangeState.laneChangeFinishing
      elif self.lane_change_state == LaneChangeState.laneChangeFinishing:
        # fade in laneline over 1s
        self.lane_change_ll_prob = min(self.lane_change_ll_prob + DT_MDL, 1.0) 

        if self.lane_change_ll_prob > 0.99: # 차선변경완료 후 1초동안 기다림. (왜?)
          self.lane_change_direction = LaneChangeDirection.none
          self.lane_change_state = LaneChangeState.off
          if one_blinker:
            self.lane_change_state = LaneChangeState.preLaneChange
            self.needTorque = True
          else:
            self.lane_change_state = LaneChangeState.off

    if self.lane_change_state in (LaneChangeState.off, LaneChangeState.preLaneChange) or nav_turn:
      self.lane_change_timer = 0.0
    else:
      self.lane_change_timer += DT_MDL

    self.prev_one_blinker = one_blinker
    self.prev_leftBlinker = carstate.leftBlinker
    self.prev_rightBlinker = carstate.rightBlinker

