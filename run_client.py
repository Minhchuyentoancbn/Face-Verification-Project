import grpc
from time import perf_counter

from app_services.register_pb2 import RegisterRequest, RegisterResponse
from app_services.register_pb2_grpc import RegisterServiceStub
from app_services.identification_pb2 import IdentificationRequest, IdentificationResponse
from app_services.identification_pb2_grpc import IdentificationServiceStub
from app_services.verification_pb2 import VerificationRequest, VerificationResponse
from app_services.verification_pb2_grpc import VerificationServiceStub


def main():
    with grpc.insecure_channel('localhost:50051') as channel:
        # stub = RegisterServiceStub(channel)
        stub = IdentificationServiceStub(channel)
        # stub = VerificationServiceStub(channel)
        start = perf_counter()
        # res: RegisterResponse = await stub.Register(RegisterRequest(user_id='test', video_path='data/test/WIN_20230517_10_37_23_Pro.mp4'))
        res: IdentificationResponse = stub.Identify(IdentificationRequest(video_path='data/test/WIN_20230517_10_37_23_Pro.mp4'))
        # res: VerificationResponse = await stub.Verify(VerificationRequest(user_id='test', video_path='data/test/WIN_20230517_10_37_23_Pro.mp4'))
        end = perf_counter()

        # Change this to whatever response you want to see
        # or comment out the print statements
        # print(f'Sucess: {res.success}')
        print(f'Time: {end - start}')

if __name__ == '__main__':
    main()
