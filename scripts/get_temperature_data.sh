rm ~/.netrc
rm ~/.urs_cookies
echo "machine urs.earthdata.nasa.gov login $URS_LOGIN password $URS_PASSWORD" >> ~/.netrc
chmod 0600 ~/.netrc
touch ~/.urs_cookies
cat ../french_temperature_data.txt | tr -d '\r' | xargs -n 1 | tac | xargs -n 1 curl -g -LJO -n -c ~/.urs_cookies -b ~/.urs_cookies