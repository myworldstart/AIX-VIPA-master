# -*- coding: utf-8 -*-
import json
from flask import Blueprint, jsonify
from datetime import datetime, time
from flask.globals import request
from app.db import valid_userId
from asm.predict import *
from PIL import Image
import ast
import io
import urllib.request
import cv2
import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from skimage.feature import hog
from app.db import db, query_d_hits, parse_row_to_dict
from asm.utils import map_docker2host

bp = Blueprint('anno_auto_tradition', __name__)

featureSize = {
    'SIFT' : 128,
    'SURF': 64,
    'ORB' : 32
}

WordCount = 300

def getImageSetmaxHW(res):  # 获取数据集中所有图像最大的Size
	maxH = 0
	maxW = 0
	for row in res:
		data = row
		imgPath = map_docker2host(data[0])
		imgPath = io.BytesIO(urllib.request.urlopen(imgPath).read())
		img = np.array(Image.open(imgPath).convert('RGB'))
		imgSize = img.shape
		if imgSize[0] > maxH:
			maxH = imgSize[0]
		if imgSize[1] > maxW:
			maxW = imgSize[1]
	return maxH, maxW

def getImageAndLabel(res):  # 获取数据集中图像以及图像label
	labelIndex = 0
	count = 0
	imgDict = dict()
	labelDict = dict()
	response = np.float32([])
	for row in res:
		data, predLabel = row
		label = ast.literal_eval(predLabel)['label'][0]
		if label not in labelDict.keys():
			labelDict[label] = labelIndex
			labelIndex += 1
		response = np.append(response, labelDict[label])
		imgPath = map_docker2host(data)
		imgPath = io.BytesIO(urllib.request.urlopen(imgPath).read())
		img = np.array(Image.open(imgPath).convert('RGB'))
		imgDict[count] = img
		count += 1
	return imgDict, labelDict, response


def extractHOG(imgDict, maxH, maxW): # 获取HOG特征
	features = []
	for i in range(len(imgDict)):
		img = imgDict[i]
		imgSize = img.shape  # 返回高和宽
		topSize = int((maxH - imgSize[0]) / 2)
		bottomSize = maxH - topSize - imgSize[0]
		leftSize = int((maxW - imgSize[1]) / 2)
		rightSize = maxW - leftSize - imgSize[1]
		replicate = cv2.copyMakeBorder(img, topSize, bottomSize, leftSize, rightSize,
									   cv2.BORDER_CONSTANT, value=0)
		gray = cv2.cvtColor(replicate, cv2.COLOR_BGR2GRAY)
		fd = hog(gray, orientations=12, block_norm='L1', pixels_per_cell=[8, 8], cells_per_block=[4, 4],
				 visualize=False,
				 transform_sqrt=True)
		features.append(fd)
	return features

# Bag of words模型，用于SIFT, SURF, ORG特征提取

def calcSiftFeature(img, featureExtract):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	if featureExtract == 'SIFT':
		sift = cv2.SIFT_create() # 设置最大特征点数目
		kp, des = sift.detectAndCompute(gray, None)
	elif featureExtract == 'SURF':
		surf = cv2.xfeatures2d.SURF_create()
		kp, des = surf.detectAndCompute(gray, None)
	elif featureExtract == 'ORB':
		orb = cv2.ORB_create()
		kp, des = orb.detectAndCompute(gray, None)
	return des

def calcFeatVec(features, centers):
	featVec = np.zeros((1, WordCount))
	for i in range(0, features.shape[0]):
		fi = features[i]
		diffMat = np.tile(fi, (WordCount, 1)) - centers
		sqSum = (diffMat**2).sum(axis=1)
		dist = sqSum**0.5
		sortedIndices = dist.argsort()
		idx = sortedIndices[0] # index of the nearest center
		featVec[0][idx] += 1
	return featVec

def initFeatureSet(featureExtract, imgDict):
	featureSet = np.float32([]).reshape(0, featureSize[featureExtract])
	count = 0
	for i in range(len(imgDict)):
		img = imgDict[i]
		des = calcSiftFeature(img, featureExtract)
		featureSet = np.append(featureSet, des, axis=0)
		count += 1
	featCnt = featureSet.shape[0]
	print(str(featCnt) + " features in " + str(count) + " images\n")
	return featureSet


def learnVocabulary(featureSet):  #生成特征词典
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)
	flags = cv2.KMEANS_RANDOM_CENTERS
	compactness, labels, centers = cv2.kmeans(featureSet, WordCount, None, criteria, 20, flags)

	# save vocabulary(a tuple of (labels, centers)) to file
	print("Done\n")
	return centers

# 分类器训练与预测

def trainClassifier(centers, featureExtract, classifer, imgDict, response):  # 训练分类器
	if featureExtract == "HOG":
		trainData = centers

	else:
		trainData = np.float32([]).reshape(0, WordCount)
		for i in range(len(imgDict)):
			img = imgDict[i]
			# img = cv2.imread(imgPath)
			features = calcSiftFeature(img, featureExtract)
			if features is None:
				continue
			featVec = calcFeatVec(features, centers)
			trainData = np.append(trainData, featVec, axis=0)

	print("图像读和标签读取完成!")
	print("开始训练分类器:")

	trainData = np.float32(trainData)
	response = response.reshape(-1, 1)

	if classifer == 'SVM':
		clf = svm.SVC(decision_function_shape='ovo')  # SVM算法
		clf.fit(trainData, response)
	elif classifer == 'KNN':
		clf = KNeighborsClassifier()  # KNN算法
		clf.fit(trainData, response)
	elif classifer == 'boosting':
		clf = AdaBoostClassifier()  # Boosting
		clf = clf.fit(trainData, response)
	elif classifer == 'DecisionTree':
		clf = tree.DecisionTreeClassifier() # 决策树算法
		clf.fit(trainData, response)
	elif classifer == 'RandomForest':
		clf = RandomForestClassifier(random_state=0)  # 随机森林
		clf.fit(trainData, response)
	elif classifer == 'NativeBayes':
		clf = MultinomialNB(alpha=2.0, fit_prior=True)  # 多项式朴素贝叶斯
		clf.fit(trainData, response)
	# clf = BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)
	# clf.fit(trainData, response)  # 伯努利朴素贝叶斯
	print("训练完成")
	return clf

def classify_single(img, featureExtract, clf, maxH, maxW, centers):  # 对单张图片进行预测
	if featureExtract == "HOG":
		imgSize = img.shape  # 返回高和宽
		topSize = int((maxH - imgSize[0]) / 2)
		bottomSize = maxH - topSize - imgSize[0]
		leftSize = int((maxW - imgSize[1]) / 2)
		rightSize = maxW - leftSize - imgSize[1]
		replicate = cv2.copyMakeBorder(img, topSize, bottomSize, leftSize, rightSize,
									   cv2.BORDER_CONSTANT, value=0)
		fd = hog(replicate, orientations=12, block_norm='L1', pixels_per_cell=[8, 8], cells_per_block=[4, 4],
				 visualize=False,
				 transform_sqrt=True)
		case = fd.reshape((1, -1)).astype(np.float64)
	else:
		features = calcSiftFeature(img, featureExtract)
		# if features is None:
		# 	continue
		featVec = calcFeatVec(features, centers)
		case = np.float32(featVec)
	label = clf.predict(case)
	return label

def isValid(featureExtract, classifer, projectID, userId):
	if not featureExtract or not classifer or not projectID or not userId:
		return False
	return valid_userId(userId)

@bp.route('/auto_tradition', methods = ['POST'])
def auto_label_train():

	print(request.json)
	projectID, userId, featureExtract, classifer = request.json.get('projectId'), request.json.get('uid'), request.json.get('feature'), request.json.get('selector')

	# projectID = "8080804e7cba54d0017cbab03a1e0000"
	# userId = "Vy7jO4QCtDm24Hbhg0EtlcibLmYo"
	# featureExtract = 'HOG'
	# classifer = 'SVM'

	print(projectID, userId, featureExtract, classifer)
	
	if not isValid(featureExtract, classifer, projectID, userId):
		print(featureExtract, classifer, projectID, userId)
		return jsonify( {'code' : 500, 'info' : 'model and projectID or model not exists'} )

	sql = "select data, predLabel from \
                (d_hits_result join d_hits on d_hits_result.hitId=d_hits.id and d_hits_result.model=d_hits.correctResult ) \
                where d_hits.projectId='{}' and d_hits_result.status='done' and d_hits_result.predLabel != '{}' and not ISNULL(d_hits_result.predLabel)".format(projectID, '{}')

	res = db.session.execute(sql)
	db.session.close()
	imgDict, labelDict, response = getImageAndLabel(res)

	maxH = None
	maxW = None
	centers = None

	if featureExtract == 'HOG':  # HOG非bag of words模型，需要单独处理
		sql = "select data from d_hits where projectId='{}'".format(projectID)
		resHW = db.session.execute(sql)
		db.session.close()
		maxH, maxW = getImageSetmaxHW(resHW)
		feature = extractHOG(imgDict, maxH, maxW)
		clf = trainClassifier(feature, featureExtract, classifer, imgDict, response)  #如果为HOG特征，传入feature

	else :
		featureSet = initFeatureSet(featureExtract, imgDict)
		centers = learnVocabulary(featureSet)
		clf = trainClassifier(centers, featureExtract, classifer, imgDict, response)

	# classify()  #对测试集进行分类，查看分类效果

	rows = query_d_hits(projectId=projectID)
	# try:
	t1 = time()
	for _, row in enumerate(rows):
		hit_dict = parse_row_to_dict(row)
		img_path = map_docker2host(hit_dict['data'])
		if 'http' in img_path:
			file = io.BytesIO(urllib.request.urlopen(img_path).read())
			img = np.array(Image.open(file).convert('RGB'))
		else:
			img = np.array(Image.open(img_path).convert('RGB'))


		label = classify_single(img, featureExtract, clf, maxH, maxW, centers)

		for l in labelDict.keys():
			if labelDict[l] == label:
				label = l

		box_info = {
			"label" : [label]
		}

		hit_result_dict = {}
		hit_result_dict['hitId'] = hit_dict['id']
		hit_result_dict['projectId'] = hit_dict['projectId']
		hit_result_dict['predLabel'] = json.dumps(box_info)
		hit_result_dict['userId'] = userId
		hit_result_dict['notes'] = 'auto-label'  # use to filter hard samples
		hit_result_dict['created_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
		hit_result_dict['updated_timestamp'] = hit_result_dict['created_timestamp']

		sql = "insert into d_hits_result " \
			  "(`hitId`, `projectId`, `result`, `userId`, `timeTakenToLabelInSec`, `notes`, `created_timestamp`, `updated_timestamp`, `predLabel`, `model`, `status`) " \
			  "values ({},'{}','{}','{}',{},'{}','{}','{}','{}','{}','{}')".format(hit_result_dict['hitId'],
																				   hit_result_dict['projectId'],
																				   [],
																				   hit_result_dict['userId'],
																				   0,
																				   hit_result_dict['notes'],
																				   hit_result_dict['created_timestamp'],
																				   hit_result_dict['updated_timestamp'],
																				   hit_result_dict['predLabel'],
																				   'human-annotation',
																				   'al')
		db.session.execute(sql)
		db.session.commit()

	t2 = time()
	print('Cost time', t2 - t1)
	return jsonify( {'code': 200, 'info': 'success'} )