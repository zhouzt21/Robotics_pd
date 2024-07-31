import os
import os
from stat import S_ISDIR, S_ISREG
from paramiko import Transport, SFTPClient

# SFTP server connection settings
hostname = 'galois.ucsd.edu'
port = 22  # default port for SFTP
username = 'sftp'
password = 'sulab666'

def get_r_portable(sftp: SFTPClient, remotedir, localdir):
    print(remotedir)
    for entry in sftp.listdir_attr(remotedir):
        remotepath = remotedir + "/" + entry.filename
        localpath = os.path.join(localdir, entry.filename)
        mode = entry.st_mode
        if S_ISDIR(mode): # type: ignore
            os.makedirs(localpath, exist_ok=True)
            get_r_portable(sftp, remotepath, localpath)
        elif S_ISREG(mode): # type: ignore
            if not os.path.exists(os.path.dirname(localpath)):
                os.makedirs(os.path.dirname(localpath), exist_ok=True)
            sftp.get(remotepath, localpath)



def download_if_not_exists(local_path, remote_path):
    """Download file from remote host if it does not exist locally."""
    if not os.path.exists(local_path):

        raise NotImplementedError(
            f"Trying to download {remote_path} to {local_path} from the SFTP server."
            " The original SFTP server is not available. Please download the file manually from"
            " https://drive.google.com/file/d/1vKvQnKYQ1vlXV-7uQSDi4ATJRQM1Y-Rc/view?usp=sharing"
            " and put it in robotics/assets"
        )

        remote_path = 'sftp/assets/' + remote_path
        print(f"Downloading {remote_path} to {local_path}...")
        import paramiko

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        sftp = None

        try:
            ssh.connect(hostname, port, username, password)
            sftp = ssh.open_sftp()
            #sftp.get(remote_path, local_path)
            get_r_portable(sftp, remote_path, local_path)
        finally:
            if sftp:
                sftp.close()
            if ssh:
                ssh.close()