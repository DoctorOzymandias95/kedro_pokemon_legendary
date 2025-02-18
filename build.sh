cd pokemon-legendary
kedro docker build
docker tag pokemon-legendary:latest pokemon-legendary:latest

echo ">> Kedro image built!"
cd ../..