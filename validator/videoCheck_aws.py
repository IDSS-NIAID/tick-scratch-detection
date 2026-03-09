import io
import os
import sys
import time
import configparser
from datetime import datetime

import streamlit as st

import s3fs
import boto3
import json
import pandas as pd
import openpyxl

__version__ = "v1.02.09"
__cfg_version__ = ":c01"
__container_version__ = ":t05"

st.set_page_config(layout="wide")
st.title("Tick Scratch Event Validator")

# environ variables
IS_PRODUCTION = os.environ.get('TVC_IS_PRODUCTION')
IS_AWS = os.environ.get('TVC_IS_AWS')

# other config
CFG_FILE = 'datacfg'
DATACFG = configparser.ConfigParser()
DATACFG.read_file(open(CFG_FILE))

def setup(cfg):
  st.session_state.data_root = cfg.get('dataroot', 'path')

  # override data root without changing the datacfg
  if len(sys.argv) == 2:
    st.session_state.data_root = sys.argv[1]

  if IS_PRODUCTION is not None:
    st.session_state.s3bucket = cfg.get('s3bucket-prod', 'bucket')
    st.session_state.s3root = cfg.get('s3bucket-prod', 'event_root')
    st.session_state.s3profile = cfg.get('s3bucket-prod', 'profile')
    st.session_state.table = cfg.get('dynamodb-prod', 'table')
    st.session_state.profile = cfg.get('dynamodb-prod', 'profile')
    st.session_state.region = cfg.get('dynamodb-prod', 'region')
  else:
    st.session_state.s3bucket = cfg.get('s3bucket', 'bucket')
    st.session_state.s3root = cfg.get('s3bucket', 'event_root')
    st.session_state.s3profile = cfg.get('s3bucket', 'profile')
    st.session_state.table = cfg.get('dynamodb', 'table')
    st.session_state.profile = cfg.get('dynamodb', 'profile')
    st.session_state.region = cfg.get('dynamodb', 'region')

  if IS_AWS is not None:
    st.session_state.s3 = s3fs.S3FileSystem()
  else:
    st.session_state.s3 = s3fs.S3FileSystem(profile=st.session_state.s3profile)

  if "currentVideoIndex" not in st.session_state:
    st.session_state.currentVideoIndex = 0
  if "currentEventNames" not in st.session_state:
    st.session_state.currentEventNames = []
  if "currentTrials" not in st.session_state:
    st.session_state.currentTrials = []
  if "numberOfConcurrentVideos" not in st.session_state:
    st.session_state.numberOfConcurrentVideos = 2
  if "eventLabels" not in st.session_state:
    st.session_state.eventLabels = {}
  if "lastModifiedDateTime" not in st.session_state:
    st.session_state.lastModifiedDateTime = ""
  if "exportData" not in st.session_state:
    st.session_state.exportData = io.BytesIO()
  if "exportDataFileName" not in st.session_state:
    st.session_state.exportDataFileName = None

  # connect to DynamoDB
  if 'client' not in st.session_state:
    connectDB()
  return

def connectDB():
  if IS_AWS is not None:
    st.session_state.session = boto3.session.Session()
  else:
    st.session_state.session = boto3.session.Session(profile_name=st.session_state.profile)

  client = st.session_state.session.client('dynamodb', region_name=st.session_state.region)
  st.session_state.client = client
  return

def createLabelIndexDict(cfg):
  st.session_state.labels = []
  st.session_state.labelIndexDict = {}

  for k, v in cfg.items('labels'):
    st.session_state.labels.append(v)

  for index, item in enumerate(st.session_state.labels):
    st.session_state.labelIndexDict[item] = index
  return

# projects/trials/events
def getProjectNames():
  s3_path = os.path.join(st.session_state.s3bucket,
                         st.session_state.s3root)
  projects = st.session_state.s3.ls(s3_path, refresh=True)
  projects = [os.path.basename(v) for v in projects if st.session_state.s3.isdir(v)]
  return projects

# projects/trials/events
def getTrialNames():
  projectPath = os.path.join(st.session_state.s3bucket, st.session_state.s3root, st.session_state.currentProject)
  trials = st.session_state.s3.ls(projectPath, refresh=True)
  trials = [os.path.basename(v) for v in trials if st.session_state.s3.isdir(v)]
  return trials

# projects/trials/events
def getEventNames():
  project = st.session_state.currentProject
  trial = st.session_state.currentTrial
  if project == None or trial == None:
    return []

  eventPath = os.path.join(st.session_state.s3bucket,
                           st.session_state.s3root,
                           project,
                           trial,
                           '*.mp4')
  movies = sorted(st.session_state.s3.glob(eventPath))
  return movies

def createDefaultEventLabels(event_names):
  labels = {}
  for event in event_names:
    key = os.path.splitext(os.path.basename(event))[0]
    labels[key] = st.session_state.labels[0]
  return labels

def writeEventLabelsToDB():
  if st.session_state.currentTrial == None:
    return

  st.session_state.lastModifiedDateTime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

  st.session_state.client.update_item(
    TableName=st.session_state.table,
    Key={"projectName": {'S': st.session_state.currentProject},
         "trialName": {'S': st.session_state.currentTrial}
         },
    UpdateExpression="set events=:e, updateTime=:t",
    ExpressionAttributeValues={
      ":e": {'S': json.dumps(st.session_state.eventLabels)},
      ":t": {'S': st.session_state.lastModifiedDateTime}
    }
  )
  return

def saveEventLabelsToDB():
  writeEventLabelsToDB()
  st.toast(" ".join(["Trial",
                    st.session_state.currentTrial,
                    "event labels saved to database",
                    st.session_state.lastModifiedDateTime]), icon=":material/check:")
  return

def getEventLabelsFromDB():
  response = st.session_state.client.query(
    TableName=st.session_state.table,
    KeyConditionExpression="projectName = :p AND trialName = :t",
    ExpressionAttributeValues={":p": {'S': st.session_state.currentProject},
                               ":t": {'S': st.session_state.currentTrial}}
  )

  if response["Count"] == 0: # records not exist
    labels = createDefaultEventLabels(st.session_state.currentEventNames)
    st.session_state.lastModifiedDateTime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    st.session_state.client.put_item(
      TableName=st.session_state.table,
      Item={
        "projectName": {'S': st.session_state.currentProject},
        "trialName": {'S': st.session_state.currentTrial},
        "events": {'S': json.dumps(labels)},
        "updateTime": {'S': st.session_state.lastModifiedDateTime}
      }
    )
    return labels
  elif response["Count"] == 1: # found the record
    eventLabels = json.loads(response["Items"][0]["events"]['S']) # AWS: you could make it easier to access query data
    return eventLabels
  else:
    st.write('\nDynamoDB query error: querying the current project/trial pair returned multiple records.')
    st.stop()

def queryEventLabelsFromDB(project, trial):
  response = st.session_state.client.query(
    TableName=st.session_state.table,
    KeyConditionExpression="projectName = :p AND trialName = :t",
    ExpressionAttributeValues={":p": {'S': project},
                               ":t": {'S': trial}}
  )

  if response["Count"] == 1:  # found the record
    eventLabels = json.loads(response["Items"][0]["events"]['S'])  # AWS: you could make it easier to access query data
    return eventLabels
  else:
    return None

def updateEventLabel(key):
  st.session_state.eventLabels[key] = st.session_state[key]
  writeEventLabelsToDB()
  return
    
def displayVideos():
  totalCount = len(st.session_state.currentEventNames)
  
  num_videos = st.session_state.numberOfConcurrentVideos
  c1_video_index = st.session_state.currentVideoIndex
  c2_video_index = c1_video_index + num_videos
  c3_video_index = c2_video_index + num_videos
  
  video_width = 0.75
  control_width = 1 - video_width
  vc1, vc2, vc3 = st.columns(3)
  with vc1:
    c1_video_index_end = c1_video_index + num_videos
    if c1_video_index_end >= totalCount:
      c1_video_index_end = totalCount
       
    for event in st.session_state.currentEventNames[c1_video_index:c1_video_index_end]:
      key = os.path.splitext(os.path.basename(event))[0]
      tokens = key.split('_')# event id, starting frame number, ending frame number

      col1, col2 = st.columns([video_width, control_width])
      with col1:
        f = st.session_state.s3.open(event, 'rb')
        st.video(f.read(), autoplay=True, loop=True)
      with col2:
        st.text(" ".join(["Event#:", tokens[0],
                            "\nStart:", tokens[1],
                            "\nEnd:", tokens[2]]))
        label = st.session_state.eventLabels[key]
        st.radio(label="Label", 
                 index=st.session_state.labelIndexDict[label], 
                 options=st.session_state.labels, 
                 key=key,
                 on_change=updateEventLabel,
                 args=[key])
  with vc2:
    if c2_video_index >= totalCount:
      return
      
    c2_video_index_end = c2_video_index + num_videos
    if c2_video_index_end >= totalCount:
      c2_video_index_end = totalCount
      
    for event in st.session_state.currentEventNames[c2_video_index:c2_video_index_end]:
      key = os.path.splitext(os.path.basename(event))[0]
      tokens = key.split('_')# event id, starting frame number, ending frame number

      col1, col2 = st.columns([video_width, control_width])
      with col1:
        f = st.session_state.s3.open(event,'rb')
        st.video(f.read(), autoplay=True, loop=True)
      with col2:
        st.text(" ".join(["Event#:", tokens[0],
                            "\nStart:", tokens[1],
                            "\nEnd:", tokens[2]]))
        label = st.session_state.eventLabels[key]
        st.radio(label="Label", 
                 index=st.session_state.labelIndexDict[label], 
                 options=st.session_state.labels, 
                 key=key,
                 on_change=updateEventLabel,
                 args=[key])
  with vc3:
    if c3_video_index >= totalCount:
      return
      
    c3_video_index_end = c3_video_index + num_videos
    if c3_video_index_end >= totalCount:
      c3_video_index_end = totalCount
      
    for event in st.session_state.currentEventNames[c3_video_index:c3_video_index_end]:
      key = os.path.splitext(os.path.basename(event))[0]
      tokens = key.split('_')# event id, starting frame number, ending frame number

      col1, col2 = st.columns([video_width, control_width])
      with col1:
        f = st.session_state.s3.open(event, 'rb')
        st.video(f.read(), autoplay=True, loop=True)
      with col2:
        st.text(" ".join(["Event#:", tokens[0],
                            "\nStart:", tokens[1],
                            "\nEnd:", tokens[2]]))
        label = st.session_state.eventLabels[key]
        st.radio(label="Label", 
                 index=st.session_state.labelIndexDict[label], 
                 options=st.session_state.labels, 
                 key=key,
                 on_change=updateEventLabel,
                 args=[key])
  return
  
def showTrialEvents():
  if st.session_state.currentTrial == None:
    return

  st.session_state.currentEventNames = getEventNames()
  if len(st.session_state.eventLabels) == 0:
    st.session_state.eventLabels = getEventLabelsFromDB()
  
  displayVideos()
  return

def showControls():
  cols = st.columns(7)
  with cols[0]:
    st.button("**<<**",
              on_click=gotoStart,
              use_container_width=True,
              disabled=st.session_state.currentProject==None)
  with cols[2]:
    st.button("**<**",
              on_click=prevPage,
              use_container_width=True,
              disabled=st.session_state.currentProject==None)
  with cols[4]:
    st.button("**\>**",
              on_click=nextPage,
              use_container_width=True,
              disabled=st.session_state.currentProject==None)
  with cols[6]:
    st.button("**\>>**",
              on_click=gotoEnd,
              use_container_width=True,
              disabled=st.session_state.currentProject==None)
  return

def prevPage():
  totalEvents = len(st.session_state.currentEventNames)
  if totalEvents == 0:
    return
  
  index = st.session_state.currentVideoIndex
  count = st.session_state.numberOfConcurrentVideos * 3 # 3 columns
  
  if index - count >= 0:
    st.session_state.currentVideoIndex -= count
  return
    
def nextPage():
  totalEvents = len(st.session_state.currentEventNames)
  if totalEvents == 0:
    return
  
  index = st.session_state.currentVideoIndex
  count = st.session_state.numberOfConcurrentVideos * 3 # 3 columns
  
  if index + count < totalEvents:
    st.session_state.currentVideoIndex += count
  return
  
def gotoStart():
  st.session_state.currentVideoIndex = 0
  return

def gotoEnd():
  totalEvents = len(st.session_state.currentEventNames)
  if totalEvents == 0:
    return

  videosPerPage = st.session_state.numberOfConcurrentVideos * 3
  numOfPages = totalEvents // videosPerPage

  if numOfPages * videosPerPage == totalEvents:
    numOfPages = numOfPages - 1
  st.session_state.currentVideoIndex = numOfPages * videosPerPage
  return

def trialSelected():
  st.session_state.currentVideoIndex = 0
  st.session_state.eventLabels = {}
  st.session_state.setAllEvents = None
  return

def setAllEvents():
  if st.session_state.currentTrial == None:
    return
  if st.session_state.setAllEvents == None:
    return
    
  for event in st.session_state.currentEventNames:
    key = os.path.splitext(os.path.basename(event))[0]
    st.session_state.eventLabels[key] = st.session_state.setAllEvents

  writeEventLabelsToDB()
  return
    
def displayStatesInfo():
  if st.session_state.currentTrial != None:
    totalCount = len(st.session_state.currentEventNames)
    end = st.session_state.currentVideoIndex + st.session_state.numberOfConcurrentVideos * 3 - 1
    if end >= totalCount:
      end = totalCount - 1
    st.caption(" ".join(["Total number of detected events:", 
                         "**:blue["+str(len(st.session_state.currentEventNames))+"]**.",
                         "Currently showing events:",
                         "**:blue["+str(st.session_state.currentVideoIndex+1)+'-'+str(end+1)+']**.',
                         "Last modified:",
                         "**:blue["+st.session_state.lastModifiedDateTime+"]**"]))
    return

def projectSelected():
  if st.session_state.currentProject == None:
    return

  st.session_state.currentTrials = getTrialNames()
  st.session_state.currentTrial = None
  
  return

def exportEventLabels():
  if st.session_state.currentProject == None:
    return

  #print('enter')
  project = st.session_state.currentProject
  trials = st.session_state.currentTrials

  sheets = []
  sheetNames = []
  for trial in trials:
    labels = queryEventLabelsFromDB(project, trial)
    if labels != None:
      start_times = []
      end_times = []
      durations = []
      values = []
      for k in labels:
        tokens = k.split("_")
        start_times.append(tokens[1])
        end_times.append(tokens[2])
        durations.append(int(tokens[2]) - int(tokens[1]))
        values.append(labels[k])
      labelDict = {'Trial_Time': start_times, 'End': end_times, 'Event_Length':durations, 'label': values}
      sheets.append(pd.DataFrame.from_dict(labelDict))
      if len(trial) > 30:
      	sheetNames.append(trial[:30])
      else:
      	sheetNames.append(trial)

  st.session_state.exportData = io.BytesIO()
  with pd.ExcelWriter(st.session_state.exportData) as writer:
    for s, sn in zip(sheets, sheetNames):
      #print(sn)
      s.index += 1 # set index to start from 1
      s.to_excel(writer, sheet_name=sn)

  st.session_state.exportData.seek(0,0)
  st.session_state.exportDataFileName = project+'.xlsx'

  #print(st.session_state.exportDataFileName)
  st.toast(" ".join(["Excel file for project",
                     "**"+st.session_state.currentProject+"**",
                     "is ready for download"]), icon=":material/check:")
  return

def downloadButtonClicked():
  st.session_state.exportDataFileName = None
  return

# initial setups
setup(DATACFG)
createLabelIndexDict(DATACFG)

# query available projects
projects = getProjectNames()

if len(projects) == 0:
  st.header("No projects found in the data root folder.")
  st.stop()

top_container = st.container()
middle_container = st.container()
info_container = st.container()
bottom_container = st.container()

with top_container:
  tcs = st.columns([0.2, 0.2, 0.1, 0.05, 0.1, 0.15, 0.1, 0.1])
  with tcs[0]:
    label = "**Projects**"
    st.selectbox(label, index=None, options=projects, key="currentProject", on_change=projectSelected)
  with tcs[1]:
    label = "**Trials**"
    st.selectbox(label, index=None, options=st.session_state.currentTrials, 
                key="currentTrial", on_change=trialSelected)
  with tcs[2]:
    st.selectbox(label="Set All Events", 
                 options=st.session_state.labels, 
                 index=None,
                 key="setAllEvents",
                 on_change=setAllEvents)
  with tcs[-3]:
    st.write("\n")
    st.button("Confirm all event labels",
              on_click=saveEventLabelsToDB,
              disabled=st.session_state.currentProject==None)
  with tcs[-2]:
    st.write("\n")
    st.button("Generate",
              on_click=exportEventLabels,
              use_container_width=True,
              disabled=st.session_state.currentProject==None)
  with tcs[-1]:
    st.write("\n")
    st.download_button("Download",
                       data=st.session_state.exportData,
                       file_name=st.session_state.exportDataFileName,
                       mime="application/octet-stream",
                       use_container_width=True,
                       on_click=downloadButtonClicked,
                       disabled=st.session_state.exportDataFileName==None)

with middle_container:
  showTrialEvents()
  
with info_container:
  displayStatesInfo()
  
with bottom_container:
  showControls()
  
st.caption(" ".join(["©2024 IAMAI/IDSS/RTB/NIAID",
                     __version__+__cfg_version__+__container_version__]))
