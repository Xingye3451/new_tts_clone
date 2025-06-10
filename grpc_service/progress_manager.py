#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import threading
from typing import Dict, Any, List, Optional


class ProgressManager:
    """任务进度管理器，用于管理任务进度信息"""

    def __init__(self):
        """初始化进度管理器"""
        self.progress_data = {}  # 存储任务进度信息
        self.lock = threading.Lock()  # 用于线程安全的锁

    def init_progress(
        self, task_id: str, progress: int = 0, message: str = "初始化中..."
    ):
        """初始化任务进度信息"""
        with self.lock:
            self.progress_data[task_id] = {
                "progress": progress,
                "status": "PROCESSING",
                "message": message,
                "is_finished": False,
                "create_time": time.time(),
                "update_time": time.time(),
            }

    def update_progress(
        self,
        task_id: str,
        progress: int,
        message: str = None,
        is_finished: bool = False,
        status: str = None,
        output_filename: str = None,
        output_path: str = None,  # 保留向后兼容
        segments: List[Dict] = None,
    ):
        """更新任务进度信息"""
        if task_id not in self.progress_data:
            return False

        with self.lock:
            data = self.progress_data[task_id]
            data["progress"] = progress
            if message:
                data["message"] = message
            if status:
                data["status"] = status
            if is_finished:
                data["is_finished"] = True
                data["status"] = "COMPLETED" if progress >= 0 else "FAILED"
            # 优先使用output_filename，向后兼容output_path
            if output_filename:
                data["output_path"] = output_filename
            elif output_path:
                data["output_path"] = output_path
            if segments:
                data["segments"] = segments
            data["update_time"] = time.time()
            return True

    def get_progress(self, task_id: str) -> Dict[str, Any]:
        """获取任务进度信息"""
        with self.lock:
            return self.progress_data.get(task_id, {}).copy()

    def check_task_exists(self, task_id: str) -> bool:
        """检查任务是否存在"""
        with self.lock:
            return task_id in self.progress_data

    def clean_old_tasks(self, max_age: int = 3600):
        """清理过期任务，默认保留1小时内的任务"""
        current_time = time.time()
        with self.lock:
            for task_id in list(self.progress_data.keys()):
                if current_time - self.progress_data[task_id]["create_time"] > max_age:
                    del self.progress_data[task_id]

    def get_all_tasks(self) -> Dict[str, Dict[str, Any]]:
        """获取所有任务的进度信息"""
        with self.lock:
            return {
                task_id: data.copy() for task_id, data in self.progress_data.items()
            }
