# import os
# from subprocess import Popen,call
# # dir = r'./Scripts/'
# a = r'Scripts\fist.bat'
# b = str(os.path.abspath(a))
# # os.system(p)
# c = "'"+ b + "'"
# print(c)
# call(a)
# # stdout, stderr = p.communicate()
# # print(p , os.getcwd())
# # os.system("C:\Windows\System32\cmd.exe /c " + a)
import pip
from subprocess import call

packages = [dist.project_name for dist in pip.get_installed_distributions()]
call("pip install --upgrade " + ' '.join(packages), shell=True)