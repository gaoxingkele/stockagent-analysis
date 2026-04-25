"""SQLAlchemy ORM 模型 - 11 张表。

导入此模块会注册所有模型到 Base.metadata, Alembic 自动迁移用。
"""
from .user import User, UserStatus
from .sms_code import SmsCode
from .transaction import PointTransaction, TransactionReason
from .job import AnalysisJob, JobStatus
from .result import AnalysisResult, AnalysisType, ResultStatus
from .progress_event import ProgressEvent
from .invite import InviteRelation, InviteMethod
from .healthcheck import HealthCheck, HealthCheckTriggerType
from .log import AppLog
from .subscription import Subscription
from .push_notification import PushNotification, PushNotificationType, PushChannel, PushStatus

__all__ = [
    "User", "UserStatus",
    "SmsCode",
    "PointTransaction", "TransactionReason",
    "AnalysisJob", "JobStatus",
    "AnalysisResult", "AnalysisType", "ResultStatus",
    "ProgressEvent",
    "InviteRelation", "InviteMethod",
    "HealthCheck", "HealthCheckTriggerType",
    "AppLog",
    "Subscription",
    "PushNotification", "PushNotificationType", "PushChannel", "PushStatus",
]
