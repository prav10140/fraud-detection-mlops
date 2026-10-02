import sys

def error_message_detail(error, error_detail: sys):
    _, _, tb = error_detail.exc_info()
    file_name = tb.tb_frame.f_code.co_filename
    return f"Error in [{file_name}] line [{tb.tb_lineno}]: {error}"

class FraudException(Exception):
    def __init__(self, error, error_detail: sys):
        super().__init__(str(error))
        self.message = error_message_detail(error, error_detail)

    def __str__(self):
        return self.message