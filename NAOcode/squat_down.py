#! /usr/bin/env python
#-*- coding: utf-8 -*-
"""
	create by Ian in 2018-9-14 21:21:06
	捡球动作v1.0
"""
from naoqi import ALProxy
import almath
import time

def stand_up():
	"""
		身体起来（抓球后）
	"""
	names = list()
	times = list()
	angles = list()
	
	# # 挺腰
	# names.append("LHipPitch")
	# times.append([2.00])
	# angles.append([-0.0000])

	# names.append("RHipPitch")
	# times.append([2.00])
	# angles.append([-0.0000])

	# 合腿
	names.append("LHipYawPitch")
	times.append([2.00])
	angles.append([0*almath.TO_RAD])

	names.append("RHipYawPitch")
	times.append([2.00])
	angles.append([0*almath.TO_RAD])

	# 站起来
	names.append("LHipPitch")
	times.append([1.00, 3.00])
	angles.append([-0.4, -0.85141])

	names.append("RHipPitch")
	times.append([1.00, 3.00])
	angles.append([-0.4, -0.85141])


	names.append("LKneePitch")
	times.append([1.00, 3.00])
	angles.append([0.7, 2.11])

	names.append("RKneePitch")
	times.append([1.00, 3.00])
	angles.append([0.7, 2.11])

	names.append("LAnklePitch")
	times.append([1.00, 3.00])
	angles.append([-0.34979, -1.16588])

	names.append("RAnklePitch")
	times.append([1.00, 3.00])
	angles.append([-0.34979, -1.16588])

	return names, times, angles

def open_leg():
	"""
		张开腿动作
	"""
	names = list()
	times = list()
	angles = list()
	# 张开腿
	names.append("LHipYawPitch")
	times.append([2.00])
	angles.append([-50*almath.TO_RAD])

	names.append("RHipYawPitch")
	times.append([2.00])
	angles.append([-50*almath.TO_RAD])

	# # 脚踝往后
	names.append("LAnklePitch")
	times.append([2.00])
	angles.append([-0.9000])

	names.append("RAnklePitch")
	times.append([2.00])
	angles.append([-0.9000])

	# 腰部往前靠
	names.append("LHipPitch")
	times.append([2.00])
	angles.append([-0.8000])

	names.append("RHipPitch")
	times.append([2.00])
	angles.append([-0.8000])
 
	# 手往前放
	names.append("LShoulderPitch")
	times.append([1.00])
	angles.append([30*almath.TO_RAD])

	names.append("RShoulderPitch")
	times.append([1.00])
	angles.append([30*almath.TO_RAD])


	names.append("LElbowRoll")
	times.append([1.00])
	angles.append([0.00])


	return names, times, angles



def hand_adjust():
	"""手部动作
	"""
	names = list()
	times = list()
	angles = list()
	

	# 
	names.append("LShoulderPitch")
	times.append([2.00])
	angles.append([41.394*almath.TO_RAD])

	names.append("LShoulderRoll")
	times.append([2.00])
	angles.append([-2.814*almath.TO_RAD])

	names.append("LElbowYaw")
	times.append([2.00])
	angles.append([-78.6655*almath.TO_RAD])

	names.append("LElbowRoll")
	times.append([2.00])
	angles.append([19.6853*almath.TO_RAD])

	names.append("LWristYaw")
	times.append([2.00])
	angles.append([104.5*almath.TO_RAD])

	return names, times, angles

def body_adjust():
	"""身体调整
	"""
	names = list()
	times = list()
	angles = list()
	

	# 脚踝后靠
	names.append("LAnklePitch")
	times.append([4.00])
	angles.append([-40*almath.TO_RAD])

	names.append("RAnklePitch")
	times.append([4.00])
	angles.append([-40*almath.TO_RAD])

	# 挺腰
	names.append("LHipYawPitch")
	times.append([6.00])
	angles.append([-40*almath.TO_RAD])

	names.append("RHipYawPitch")
	times.append([6.00])
	angles.append([-40*almath.TO_RAD])


	# 下弯腰
	names.append("LHipPitch")
	times.append([6.00])
	angles.append([-85*almath.TO_RAD])

	names.append("RHipPitch")
	times.append([6.00])
	angles.append([-85*almath.TO_RAD])


	# 手伸直
	names.append("LShoulderRoll")
	times.append([4.00])
	angles.append([-5*almath.TO_RAD])

	names.append("RShoulderRoll")
	times.append([4.00])
	angles.append([-5*almath.TO_RAD])

	return names, times, angles


def squat_down():
	"""
		蹲下动作
	"""
	names = list()
	times = list()
	angles = list()
	names.append("LHipPitch")
	times.append([1.00, 3.00])
	angles.append([-0.4, -0.85141])

	names.append("RHipPitch")
	times.append([1.00, 3.00])
	angles.append([-0.4, -0.85141])


	names.append("LKneePitch")
	times.append([1.00, 3.00])
	angles.append([0.7, 2.11])

	names.append("RKneePitch")
	times.append([1.00, 3.00])
	angles.append([0.7, 2.11])

	names.append("LAnklePitch")
	times.append([1.00, 3.00])
	angles.append([-0.34979, -1.16588])

	names.append("RAnklePitch")
	times.append([1.00, 3.00])
	angles.append([-0.34979, -1.16588])


	return names, times, angles


def per_run(IP):
	# 慎用，仅用于第一次使用
	life = ALProxy("ALAutonomousLife", IP, 9559)
	print "good"
	life.setState('disabled') #关闭自主生活模式


def run(IP):
	motion = ALProxy("ALMotion", IP, 9559)
	motion.wakeUp() # 唤醒机器人
	motion.moveInit() # 初始化行走动作
	
	# 蹲下
	names, times, angles = squat_down()
	motion.angleInterpolationBezier(names, times, angles)

	# 张腿
	print "open leg"
	names, times, angles = open_leg()
	motion.angleInterpolationBezier(names, times, angles)

	# 微调
	motion.openHand('LHand')
	time.sleep(1)

	print "hand_adjust"
	names, times, angles = hand_adjust()
	motion.angleInterpolationBezier(names, times, angles)

	print "body_adjust"
	names, times, angles = body_adjust()
	motion.angleInterpolationBezier(names, times, angles)


	name = ["LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll", "LWristYaw"]
	stiffnesses    = 0.0
	for i in name:
		commandAngles = motion.setStiffnesses(i, stiffnesses)

		
	time.sleep(20)

	name = ["LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll", "LWristYaw"]
	useSensors    = False

	for i in name:
		commandAngles = motion.getAngles(i, useSensors)
		print i + "   angles:"
		print str(commandAngles[0] / almath.TO_RAD)
		print ""
	

	# 抓球
	
	motion.closeHand('LHand')

	# # 站起来
	# names, times, angles = stand_up()
	# motion.angleInterpolationBezier(names, times, angles)

if __name__ == '__main__':
	ip = "192.168.1.102"
	#per_run(ip)
	run(ip)