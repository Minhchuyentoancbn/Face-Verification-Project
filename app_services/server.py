import asyncio
import grpc
from .services import register, verification, identification

from .register_pb2_grpc import RegisterServiceServicer, add_RegisterServiceServicer_to_server
from .identification_pb2_grpc import IdentificationServiceServicer, add_IdentificationServiceServicer_to_server
from .verification_pb2_grpc import VerificationServiceServicer, add_VerificationServiceServicer_to_server

from .register_pb2 import RegisterRequest, RegisterResponse
from .identification_pb2 import IdentificationRequest, IdentificationResponse
from .verification_pb2 import VerificationRequest, VerificationResponse


class RegisterService(RegisterServiceServicer):
    async def Register(self, request: RegisterRequest, context) -> RegisterResponse:
        print('Received Register request')
        print("RegisterService: Register")
        userid = request.user_id
        video_path = request.video_path
        success = await register(video_path=video_path, userid=userid)
        return RegisterResponse(success=success)
    

class IdentificationService(IdentificationServiceServicer):
    async def Identify(self, request: IdentificationRequest, context) -> IdentificationResponse:
        print('Received Identify request')
        print("IdentificationService: Identify")
        video_path = request.video_path
        user_id = await identification(video_path=video_path)
        return IdentificationResponse(user_id=user_id)
    
    
class VerificationService(VerificationServiceServicer):
    async def Verify(self, request: VerificationRequest, context) -> VerificationResponse:
        print('Received Verify request')
        print("VerificationService: Verify")
        video_path = request.video_path
        user_id = request.user_id
        verfied, similarity = await verification(video_path=video_path, userid=user_id)
        return VerificationResponse(verified=verfied, similarity=similarity)
    

async def serve():
    server = grpc.aio.server()
    add_RegisterServiceServicer_to_server(RegisterService(), server)
    add_IdentificationServiceServicer_to_server(IdentificationService(), server)
    add_VerificationServiceServicer_to_server(VerificationService(), server)
    listen_addr = '[::]:50051'
    server.add_insecure_port(listen_addr)
    print(f'Starting server on {listen_addr}')
    await server.start()
    await server.wait_for_termination()