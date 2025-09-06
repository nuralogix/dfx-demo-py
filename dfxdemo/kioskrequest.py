import libdfx as dfxsdk
from quart_schema import QuartSchema, validate_request
from pydantic import BaseModel 
from typing import Optional
from typing import Union


class KioskRequest(BaseModel):
    deviceId: str
    requestId: str
    gender: Optional[str] = None
    age: Union[int , float , None] = None
    height: Union[int , float , None] = None
    weight: Union[int , float , None] = None
    smoking: Optional[int] = None
    diabetes: Optional[str] = None
    bloodpressuremedication: Optional[int] = None
    hypertensive: Optional[int] = None

    def get_demographics(self):
        return {
            "gender": self.gender,
            "age": self.age,
            "height": self.height,
            "weight": self.weight,
            "smoking": self.smoking,
            "diabetes": self.diabetes,
            "bloodpressuremedication": self.bloodpressuremedication,
            "hypertensive": self.hypertensive,
        }

    def log_user_demographics(self):
        if (gender := self.gender) is not None:
            sex_at_birth = dfxsdk.FaceValue.SEX_NOT_PROVIDED
            if gender == "male":
                sex_at_birth = dfxsdk.FaceValue.SEX_ASSIGNED_MALE_AT_BIRTH
            elif gender == "female":
                sex_at_birth = dfxsdk.FaceValue.SEX_ASSIGNED_FEMALE_AT_BIRTH
            print("       SEX_ASSIGNED_AT_BIRTH:", sex_at_birth)
        else:
            print("       Warn: SEX_ASSIGNED_AT_BIRTH not provided")

        if isinstance(age := self.age, (int, float)):
            print(f"       AGE_YEARS: {age}")
        else:
            print("       Warn: AGE_YEARS not provided")

        if isinstance(height := self.height, (int, float)):
            print(f"       HEIGHT_CM: {height}")
        else:
            print("       Warn: HEIGHT_CM not provided")

        if isinstance(weight := self.weight, (int, float)):
            print(f"       WEIGHT_KG: {weight}")
        else:
            print("       Warn: WEIGHT_KG not provided")

        if (smoking := self.smoking) in (0, 1):
            print(f"       SMOKER: {smoking}")
        else:
            print("       Warn: SMOKER not provided")

        if (diabetes_text := self.diabetes) in ["0", "type1", "type2"]:
            diabetes = dfxsdk.FaceValue.DIABETES_NONE
            if diabetes_text == "type1":
                diabetes = dfxsdk.FaceValue.DIABETES_TYPE1
            elif diabetes_text == "type2":
                diabetes = dfxsdk.FaceValue.DIABETES_TYPE2
            print("       DIABETES:", diabetes)
        else:
            print("       Warn: DIABETES not provided")

        if (bloodpressuremedication := self.bloodpressuremedication) in (0, 1):
            print(f"       BLOOD_PRESSURE_MEDICATION: {bloodpressuremedication}")
        else:
            print("       Warn: BLOOD_PRESSURE_MEDICATION not provided")

        if (hypertensive := self.hypertensive) in (0, 1):
            print(f"       HYPERTENSIVE: {hypertensive}")
        else:
            print("       Warn: HYPERTENSIVE not provided")

        return True
