CXX_FLAGS=-I/home/aaron/.local/include -std=c++11 -I /home/aaron/Softs/hadoop-2.7.2/include -I /home/aaron/Softs/mesos-0.26.0/include -I /home/aaron/Softs/mesos-0.26.0/build/include -I ../../include
LD_FLAGS=-L/home/aaron/.local/lib -L /home/aaron/Projects/incubator-singa/.libs -L/home/aaron/Softs/mesos-0.26.0/build/lib -L/home/aaron/Softs/hadoop-2.7.2/lib/native $(JAVA_HOME)/jre/lib/amd64/server/libjvm.so -lmesos -lsinga -lhdfs -lprotobuf -lglog
EXE=scheduler
OBJS=singa_scheduler.o scheduler.pb.o
PROTOS=scheduler.proto
PROTO_HDRS=scheduler.pb.h
PROTO_SRCS=scheduler.pb.cc

CXX=g++

all: $(PROTO_HDRS) $(EXE)

$(PROTO_SRCS) $(PROTO_HDRS): $(PROTOS)
	/home/aaron/.local/bin/protoc --cpp_out=. $(PROTOS)

$(EXE): $(OBJS)
	$(CXX) -o $@ $(OBJS) $(LD_FLAGS)
	rm -rf *.o

%.o: %.cc
	$(CXX) $(CXX_FLAGS) -c $<     

clean: 
	rm -rf *.o $(EXE) $(PROTO_HDRS) $(PROTO_SRCS)

