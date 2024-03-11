#include <memory>
#include <string>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include "ggml-rpc.h"

int main(int argc, char * argv[])
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <port>\n", argv[0]);
        return 1;
    }
    int port = std::stoi(argv[1]);
    std::string server_address = "0.0.0.0:" + std::to_string(port);
    BackendImpl service;

    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    grpc::ServerBuilder builder;
    // Listen on the given address without any authentication mechanism.
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    builder.SetMaxSendMessageSize(1024 * 1024 * 1024);
    builder.SetMaxMessageSize(1024 * 1024 * 1024);
    builder.SetMaxReceiveMessageSize(1024 * 1024 * 1024);
    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
    std::cout << "RPC backend listening on " << server_address << std::endl;

    // Wait for the server to shutdown. Note that some other thread must be
    // responsible for shutting down the server for this call to ever return.
    server->Wait();

    return 0;
}
