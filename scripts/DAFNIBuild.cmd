
del build_dafni\*.yaml
del build_dafni\*.tar
del build_dafni\*.tar.gz
docker build -t pyquant3:latest ./src
docker save -o build_dafni/pyquant3.tar pyquant3:latest
c:\"Program Files"\7-Zip\7z.exe a -tgzip build_dafni\pyquant3.tar.gz build_dafni\pyquant3.tar
del build_dafni\pyquant3.tar
copy src\dafni-model-definition.yaml build_dafni
