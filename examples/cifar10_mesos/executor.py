#!/usr/bin/env python

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import threading
import time
import glob
from multiprocessing import Process

MESOS_ROOT="/home/aaron/Softs/mesos-0.27.0/build"
for egg in glob.glob(os.path.join(MESOS_ROOT,'src','python','dist','*.egg')):
    sys.path.append(os.path.abspath(egg))


import mesos.interface
from mesos.interface import mesos_pb2
import mesos.native

import main

class MyExecutor(mesos.interface.Executor):
    def launchTask(self, driver, task):
        # Create a thread to run the task. Tasks should always be run in new
        # threads or processes, rather than inside launchTask itself.

	def runTask(driver,task):
            # We are in the child.
            # This is where one would perform the requested task.

            try:
		print "test"
                sys.argv.append("-singa_conf")
		sys.argv.append("/home/aaron/Projects/incubator-singa/conf/singa.conf")
                model = main.buildModel(1)
                main.product(model)
            except Exception as e:
		print str(e)
                print "Sending status failed..."
                update = mesos_pb2.TaskStatus()
                update.task_id.value = task.task_id.value
                update.state = mesos_pb2.TASK_FAILED
                update.data = 'data with a \0 byte' 
                driver.sendStatusUpdate(update)
                print "Sent status failed"
                sys.exit(1) 
            print "Sending status finished..."
            update = mesos_pb2.TaskStatus()
            update.task_id.value = task.task_id.value
            update.state = mesos_pb2.TASK_FINISHED
            update.data = 'data with a \0 byte'
            driver.sendStatusUpdate(update)
            print "Sent status finished"
        print "Running task %s" % task.task_id.value
        update = mesos_pb2.TaskStatus()
        update.task_id.value = task.task_id.value
        update.state = mesos_pb2.TASK_RUNNING
        update.data = 'data with a \0 byte'
        driver.sendStatusUpdate(update)
        time.sleep(20)
        p = Process(target=runTask, args=(driver,task))
        p.start()
        p.join()
        '''
        pid = os.fork()
        if pid == 0:
            # We are in the child.
            # This is where one would perform the requested task.

            try:
                model = main.buildModel(1)
		print "test"
                sys.argv.append("-singa_conf")
		sys.argv.append("/home/aaron/Projects/incubator-singa/conf/singa.conf")
                main.product(model)
            except Exception as e:
		print str(e)
                print "Sending status failed..."
                update = mesos_pb2.TaskStatus()
                update.task_id.value = task.task_id.value
                update.state = mesos_pb2.TASK_FAILED
                update.data = 'data with a \0 byte' 
                driver.sendStatusUpdate(update)
                print "Sent status failed"
                sys.exit(1) 
            print "Sending status finished..."
            update = mesos_pb2.TaskStatus()
            update.task_id.value = task.task_id.value
            update.state = mesos_pb2.TASK_FINISHED
            update.data = 'data with a \0 byte'
            driver.sendStatusUpdate(update)
            print "Sent status finished"
            sys.exit(0)
	else:
            # in parent 
            print "Running task %s" % task.task_id.value
            update = mesos_pb2.TaskStatus()
            update.task_id.value = task.task_id.value
            update.state = mesos_pb2.TASK_RUNNING
            update.data = 'data with a \0 byte'
            driver.sendStatusUpdate(update)
            time.sleep(20)
        '''
    def frameworkMessage(self, driver, message):
        # Send it back to the scheduler.
        print "send message"
        driver.sendFrameworkMessage(message)

if __name__ == "__main__":
    print "Starting executor on slave"
    driver = mesos.native.MesosExecutorDriver(MyExecutor())
    sys.exit(0 if driver.run() == mesos_pb2.DRIVER_STOPPED else 1)
