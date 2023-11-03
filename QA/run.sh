echo start
nohup bash -c "\
(nohup python qa_train.py --lang english --gpu 0 ) & \
(nohup python qa_train.py --lang bengail --gpu 1 ) & \
(nohup python qa_train.py --lang arabic --gpu 2 ) & \
(nohup python qa_train.py --lang finnish --gpu 3 ) & \
wait && \
(echo ended1)"\
&& \
nohup bash -c "\
(nohup python qa_train.py --lang indonesian --gpu 0 ) & \
(nohup python qa_train.py --lang russian --gpu 1 ) & \
(nohup python qa_train.py --lang swahili --gpu 2 ) & \
(nohup python qa_train.py --lang korean --gpu 3 ) & \
(echo ended1)"\
