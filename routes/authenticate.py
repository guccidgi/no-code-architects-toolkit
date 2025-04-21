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



from flask import Blueprint, request, jsonify, current_app
from app_utils import *
from functools import wraps
import os
import logging

auth_bp = Blueprint('auth', __name__)

API_KEY = os.environ.get('API_KEY')

@auth_bp.route('/authenticate', methods=['GET'])
@queue_task_wrapper(bypass_queue=True)
def authenticate_endpoint(**kwargs):
    # u6aa2u67e5u591au7a2eu53efu80fdu7684 API u91d1u9470 header u683cu5f0f
    api_key = request.headers.get('X-API-Key') or request.headers.get('x-api-key') or request.headers.get('X-API-KEY')
    
    if api_key and api_key.lower() == API_KEY.lower():
        return "Authorized", "/authenticate", 200
    else:
        logging.warning(f"Authentication failed - API key mismatch")
        return "Authorization failed - please check your credentials\nUnauthorized", "/authenticate", 401
