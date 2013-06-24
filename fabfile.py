"""
    REQUIREMENTS:
        - install pip with distribute (http://packages.python.org/distribute/)
        - sudo pip install Fabric

"""

from fabric.api import local

def run():
    """ start the local app server """
    local("python 'Run the application'")

def commit(m="Fab-update the app"):
    """ save the to github """
    local("git add .")
    local("git commit -a -m '{0}'".format(m))
    local("git push")
