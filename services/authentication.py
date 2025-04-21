# Copyright (c) 2025 Stephen G. Pope
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.



from functools import wraps
from flask import request, jsonify
from config import API_KEY
import logging

def authenticate(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 檢查多種可能的 API 金鑰 header 格式
        api_key = request.headers.get('X-API-Key') or request.headers.get('x-api-key') or request.headers.get('X-API-KEY')
        
        if not api_key or api_key.lower() != API_KEY.lower():
            logging.warning(f"Authentication failed - API key mismatch or missing")
            return jsonify({"message": "Authorization failed - please check your credentials"}), 401
        return func(*args, **kwargs)
    return wrapper
