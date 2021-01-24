# this file is only used to run the API on an Apache server with WSGI_mod
import sys
sys.path.insert(0, '/var/www/html/sotags')

from sotags import app as application
