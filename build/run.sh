export DATA_DIR="/media/jbl/Secondary SSD/melytanulas_hf/data/"
export OUT_DIR="/media/jbl/Secondary SSD/melytanulas_hf/out/"
export CODE_DIR="/media/jbl/Secondary SSD/melytanulas_hf/lib/"

docker run \
  --gpus all \
  -v "${DATA_DIR}:/data" \
  -v "${OUT_DIR}:/out" \
  -v "${CODE_DIR}:/code" \
  melytanulas:latest \
  /bin/bash startup.sh
