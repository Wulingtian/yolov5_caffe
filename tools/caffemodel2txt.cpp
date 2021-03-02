#include <string>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using namespace caffe;
using std::string;
using google::protobuf::Message;


int main(int argc, char** argv) {
	
	if(4 != argc) {
		printf("Usage:\n\t%s c2t[t2c] caffemodel_path[txt_path] txt_path[caffemodel_path]\n", argv[0]);
		return 0;
	}
	
	string flag = string(argv[1]);
	string file1 = string(argv[2]);
	string file2 = string(argv[3]);
	NetParameter proto;
	if(string("c2t") == flag) {		
		if(ReadProtoFromBinaryFile(file1.c_str(),&proto)) {
			WriteProtoToTextFile(proto,file2);
		} else {
			printf("Read %s failed!\n",file1.c_str());
			return -1;
		}
	} else 	if(string("t2c") == flag) {		
		if(ReadProtoFromTextFile(file1.c_str(),&proto)) {
			WriteProtoToBinaryFile(proto,file2);
		} else {
			printf("Read %s failed!\n",file1.c_str());
			return -1;
		}			
	}
	
	return 0;
}
