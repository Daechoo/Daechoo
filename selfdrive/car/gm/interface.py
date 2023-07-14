#!/usr/bin/env python3
from typing import List
from cereal import car
from math import fabs, exp
from panda import Panda

from common.conversions import Conversions as CV
from common.realtime import sec_since_boot
from selfdrive.car import STD_CARGO_KG, create_button_event, scale_tire_stiffness, get_safety_config
from selfdrive.car.gm.radar_interface import RADAR_HEADER_MSG
from selfdrive.car.gm.values import CAR, AccState, CruiseButtons, CarControllerParams, EV_CAR, CAMERA_ACC_CAR, CanBus, CC_ONLY_CAR
from selfdrive.car.interfaces import CarInterfaceBase, TorqueFromLateralAccelCallbackType, FRICTION_THRESHOLD
from selfdrive.controls.lib.drive_helpers import get_friction

ButtonType = car.CarState.ButtonEvent.Type
EventName = car.CarEvent.EventName
ENABLE_BUTTONS = {CruiseButtons.RES_ACCEL, CruiseButtons.DECEL_SET, CruiseButtons.CANCEL}
GearShifter = car.CarState.GearShifter
TransmissionType = car.CarParams.TransmissionType
NetworkLocation = car.CarParams.NetworkLocation
BUTTONS_DICT = {CruiseButtons.RES_ACCEL: ButtonType.accelCruise, CruiseButtons.DECEL_SET: ButtonType.decelCruise,
                CruiseButtons.MAIN: ButtonType.altButton3, CruiseButtons.CANCEL: ButtonType.cancel}


class CarInterface(CarInterfaceBase):

  def _update(self, c: car.CarControl) -> car.CarState:
    pass

  @staticmethod
  def get_pid_accel_limits(CP, current_speed, cruise_speed):
    return CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX

  # Determined by iteratively plotting and minimizing error for f(angle, speed) = steer.
  @staticmethod
  def get_steer_feedforward_volt(desired_angle, v_ego):
    desired_angle *= 0.02904609
    sigmoid = desired_angle / (1 + fabs(desired_angle))
    return 0.10006696 * sigmoid * (v_ego + 3.12485927)

  @staticmethod
  def get_steer_feedforward_acadia(desired_angle, v_ego):
    desired_angle *= 0.09760208
    sigmoid = desired_angle / (1 + fabs(desired_angle))
    return 0.04689655 * sigmoid * (v_ego + 10.028217)

  def get_steer_feedforward_function(self):
    if self.CP.carFingerprint == CAR.VOLT2018:
      return self.get_steer_feedforward_volt
    elif self.CP.carFingerprint == CAR.ACADIA:
      return self.get_steer_feedforward_acadia
    else:
      return CarInterfaceBase.get_steer_feedforward_default

  @staticmethod
  def torque_from_lateral_accel_bolt(lateral_accel_value: float, torque_params: car.CarParams.LateralTorqueTuning,
                                     lateral_accel_error: float, lateral_accel_deadzone: float, friction_compensation: bool) -> float:
    friction = get_friction(lateral_accel_error, lateral_accel_deadzone, FRICTION_THRESHOLD, torque_params, friction_compensation)

    def sig(val):
      return 1 / (1 + exp(-val)) - 0.5

    # The "lat_accel vs torque" relationship is assumed to be the sum of "sigmoid + linear" curves
    # An important thing to consider is that the slope at 0 should be > 0 (ideally >1)
    # This has big effect on the stability about 0 (noise when going straight)
    # ToDo: To generalize to other GMs, explore tanh function as the nonlinear
    a, b, c, _ = [2.6531724862969748, 1.0, 0.1919764879840985, 0.009054123646805178]  # weights computed offline

    steer_torque = (sig(lateral_accel_value * a) * b) + (lateral_accel_value * c)
    return float(steer_torque) + friction

  def torque_from_lateral_accel(self) -> TorqueFromLateralAccelCallbackType:
    if self.CP.carFingerprint == CAR.BOLT_EUV:
      return self.torque_from_lateral_accel_bolt
    else:
      return self.torque_from_lateral_accel_linear

  @staticmethod
  def _get_params(ret, candidate, fingerprint, car_fw, experimental_long):
    ret.carName = "gm"
    ret.safetyConfigs = [get_safety_config(car.CarParams.SafetyModel.gm)]
    ret.autoResumeSng = False

    # ret.enableGasInterceptor = 0x201 in fingerprint[0]
    ret.enableGasInterceptor = 512 in fingerprint[0]

    if candidate in EV_CAR:
      ret.transmissionType = TransmissionType.direct
    else:
      ret.transmissionType = TransmissionType.automatic

    ret.longitudinalTuning.deadzoneBP = [0.]
    ret.longitudinalTuning.deadzoneV = [0.15]

    ret.longitudinalTuning.kpBP = [0.]
    ret.longitudinalTuning.kiBP = [0.]

    if candidate in CAMERA_ACC_CAR:
      ret.experimentalLongitudinalAvailable = candidate not in CC_ONLY_CAR
      ret.networkLocation = NetworkLocation.fwdCamera
      ret.radarUnavailable = True  # no radar
      ret.pcmCruise = True
      ret.safetyConfigs[0].safetyParam |= Panda.FLAG_GM_HW_CAM
      ret.minEnableSpeed = 5 * CV.KPH_TO_MS
      ret.minSteerSpeed = 10 * CV.KPH_TO_MS

      # Tuning for experimental long
      ret.longitudinalTuning.kpV = [1.5]
      ret.longitudinalTuning.kiV = [0.72]
      ret.stoppingDecelRate = 2.0  # reach brake quickly after enabling
      ret.vEgoStopping = 0.25
      ret.vEgoStarting = 0.25

      if experimental_long:
        ret.pcmCruise = False
        ret.openpilotLongitudinalControl = True
        ret.safetyConfigs[0].safetyParam |= Panda.FLAG_GM_HW_CAM_LONG

    else:  # ASCM, OBD-II harness
      ret.openpilotLongitudinalControl = True
      ret.networkLocation = NetworkLocation.gateway
      ret.radarUnavailable = False # RADAR_HEADER_MSG not in fingerprint[CanBus.OBSTACLE] and not use_off_car_defaults
      ret.pcmCruise = False  # stock non-adaptive cruise control is kept off
      # supports stop and go, initial engage could (conservatively) be above -1mph
      ret.minEnableSpeed = -1
      ret.minSteerSpeed = 3.0 * CV.MPH_TO_MS

      # Tuning
      ret.longitudinalTuning.kpV = [2.0]
      ret.longitudinalTuning.kiV = [0.36]
      ret.safetyConfigs[0].safetyParam |= Panda.FLAG_GM_HW_CAM_LONG

    # These cars have been put into dashcam only due to both a lack of users and test coverage.
    # These cars likely still work fine. Once a user confirms each car works and a test route is
    # added to selfdrive/car/tests/routes.py, we can remove it from this list.
    ret.dashcamOnly = candidate in {CAR.CADILLAC_ATS, CAR.HOLDEN_ASTRA, CAR.MALIBU, CAR.BUICK_REGAL, CAR.EQUINOX}



    # for autohold on ui icon
    # ret.enableAutoHold = 241 in fingerprint[0]
    # ret.openpilotLongitudinalControl = True
    # Start with a baseline tuning for all GM vehicles. Override tuning as needed in each model section below.
    ret.steerActuatorDelay = 0.2  # Default delay, not measured yet
    tire_stiffness_factor = 0.444  # not optimized yet

    ret.steerLimitTimer = 0.6
    ret.radarTimeStep = 0.0667  # GM radar runs at 15Hz instead of standard 20Hz


    if candidate == CAR.VOLT2018:
      ret.lateralTuning.init('torque')
      ret.minEnableSpeed = -1 * CV.MPH_TO_MS
      ret.mass = 1607. + STD_CARGO_KG
      ret.wheelbase = 2.69
      ret.steerRatio = 17.7  # Stock 15.7, LiveParameters
      tire_stiffness_factor = 0.469  # Stock Michelin Energy Saver A/S, LiveParameters
      ret.centerToFront = ret.wheelbase * 0.45  # Volt Gen 1, TODO corner weigh

      CarInterfaceBase.configure_torque_tune(candidate, ret.lateralTuning)
      ret.steerActuatorDelay = 0.2
      
      ret.longitudinalTuning.kpBP = [0.]
      ret.longitudinalTuning.kpV = [2.4]
      ret.longitudinalTuning.kiBP = [0.]
      ret.longitudinalTuning.kiV = [0.35]

    # TODO: start from empirically derived lateral slip stiffness for the civic and scale by
    # mass and CG position, so all cars will have approximately similar dyn behaviors
    ret.tireStiffnessFront, ret.tireStiffnessRear = scale_tire_stiffness(ret.mass, ret.wheelbase, ret.centerToFront,
                                                                         tire_stiffness_factor=tire_stiffness_factor)

    ret.steerControlType = car.CarParams.SteerControlType.torque
    ret.stoppingControl = True
    ret.startingState = True

    ret.longitudinalActuatorDelayLowerBound = 0.15
    ret.longitudinalActuatorDelayUpperBound = 0.25
    ret.longitudinalTuning.kf = 1.0
    ret.stoppingDecelRate = 3.0  # reach stopping target smoothly, brake_travel/s while trying to stop
    ret.stopAccel = -2.0  # Required acceleration to keep vehicle stationary
    ret.vEgoStopping = 0.35  # Speed at which the car goes into stopping state, when car starts requesting stopping accel
    ret.vEgoStarting = 0.25  # Speed at which the car goes into starting state, when car starts requesting starting accel,

    return ret

  # returns a car.CarState
  def update(self, c: car.CarControl, can_strings: List[bytes]) -> car.CarState:
    self.cp.update_strings(can_strings)
    self.cp_cam.update_strings(can_strings)
    self.cp_loopback.update_strings(can_strings) # GM: EPS fault workaround (#22404)

    self.cp_chassis.update_strings(can_strings) # for Brake Light
    ret = self.CS.update(self.cp, self.cp_cam, self.cp_loopback, self.cp_chassis) # GM: EPS fault workaround (#22404)


    cruiseEnabled = self.CS.pcm_acc_status != AccState.OFF
    ret.cruiseState.enabled = cruiseEnabled

    ret.canValid = self.cp.can_valid and self.cp_cam.can_valid and self.cp_loopback.can_valid # GM: EPS fault workaround (#22404)
    ret.engineRpm = self.CS.engineRPM

    buttonEvents = []

    if self.CS.cruise_buttons != self.CS.prev_cruise_buttons and self.CS.prev_cruise_buttons != CruiseButtons.INIT:
      buttonEvents = [create_button_event(self.CS.cruise_buttons, self.CS.prev_cruise_buttons, BUTTONS_DICT, CruiseButtons.UNPRESS)]
      # Handle ACCButtons changing buttons mid-press
      if self.CS.cruise_buttons != CruiseButtons.UNPRESS and self.CS.prev_cruise_buttons != CruiseButtons.UNPRESS:
        buttonEvents.append(create_button_event(CruiseButtons.UNPRESS, self.CS.prev_cruise_buttons, BUTTONS_DICT, CruiseButtons.UNPRESS))


    ret.buttonEvents = buttonEvents

    if cruiseEnabled and self.CS.lka_button and self.CS.lka_button != self.CS.prev_lka_button:
      self.CS.lkMode = not self.CS.lkMode

    if self.CS.distance_button and self.CS.distance_button != self.CS.prev_distance_button:
       self.CS.follow_level -= 1
       if self.CS.follow_level < 1:
         self.CS.follow_level = 3

    ret.cruiseGap = self.CS.follow_level

    # The ECM allows enabling on falling edge of set, but only rising edge of resume
    events = self.create_common_events(ret, extra_gears=[GearShifter.sport, GearShifter.low,
                                                         GearShifter.eco, GearShifter.manumatic],
                                       pcm_enable=False, enable_buttons=(ButtonType.decelCruise, ButtonType.accelCruise))

    if not self.CP.pcmCruise:
      if any(b.type == ButtonType.decelCruise and b.pressed for b in ret.buttonEvents):
        events.add(EventName.buttonEnable)
      if any(b.type == ButtonType.accelCruise and b.pressed for b in ret.buttonEvents):
        events.add(EventName.buttonEnable)
    below_min_enable_speed = ret.vEgo < self.CP.minEnableSpeed or self.CS.moving_backward
    if below_min_enable_speed and not (ret.standstill and ret.brake >= 20 and
                                       self.CP.networkLocation == NetworkLocation.fwdCamera):
      events.add(EventName.belowEngageSpeed)

    #autohold event
    if self.CS.autoHoldActivated:
      events.add(EventName.autoHold)

    ret.events = events.to_msg()

    # copy back carState packet to CS
    self.CS.out = ret.as_reader()

    return self.CS.out

    events = self.create_common_events(ret, extra_gears=[GearShifter.sport, GearShifter.low,
                                                         GearShifter.eco, GearShifter.manumatic],
                                       pcm_enable=False, enable_buttons=(ButtonType.decelCruise, ButtonType.accelCruise))


    if not self.CP.pcmCruise:
      if any(b.type == ButtonType.decelCruise and b.pressed for b in ret.buttonEvents):
        events.add(EventName.buttonEnable)
      if any(b.type == ButtonType.accelCruise and b.pressed for b in ret.buttonEvents):
        events.add(EventName.buttonEnable)
    below_min_enable_speed = ret.vEgo < self.CP.minEnableSpeed or self.CS.moving_backward
    if below_min_enable_speed and not (ret.standstill and ret.brake >= 20 and
                                       self.CP.networkLocation == NetworkLocation.fwdCamera):
      events.add(EventName.belowEngageSpeed)
    if ret.cruiseState.standstill:
      events.add(EventName.resumeRequired)

    t = sec_since_boot()
    if self.CS.autoHoldActivated:
      self.CS.lastAutoHoldTime = t
    if EventName.accFaulted in events.events and \
        (t - self.CS.sessionInitTime < 10.0 or
        t - self.CS.lastAutoHoldTime < 1.0):
      events.events.remove(EventName.accFaulted)

    # autohold on ui icon
    if self.CS.autoHoldActivated == True:
      ret.brakeHoldActive = True

    if self.CS.autoHoldActivated == False:
      ret.brakeHoldActive = False
    ret.events = events.to_msg()

    # copy back carState packet to CS
    self.CS.out = ret.as_reader()

    return self.CS.out

  def apply(self, c):
    can_sends = self.CC.update(c, self.CS)
    # Release Auto Hold and creep smoothly when regenpaddle pressed
    if self.CS.regenPaddlePressed and self.CS.autoHold:
      self.CS.autoHoldActive = False

    if self.CS.autoHold and not self.CS.autoHoldActive and not self.CS.regenPaddlePressed:
      if self.CS.out.vEgo > 0.03:
        self.CS.autoHoldActive = True
      elif self.CS.out.vEgo < 0.02 and self.CS.out.brakePressed:
        self.CS.autoHoldActive = True
    return can_sends
