# CHANGELOG

## v0.3.7 (2024-11-01)

### Fix

* Fix(spikeglx): robust IMAX value detection from IMEC file (metadata 3.0)

## v0.3.6 (2024-10-01)

### Chore

* chore(.github): add new GitHub Action callers for devcontainer, mkdocs, and semantic release ([`fc8ac1d`](https://github.com/datajoint/element-array-ephys/commit/fc8ac1d8159e07714f1041bca67fa3546451f1a0))

### Fix

* fix(spikeglx): minor bugfix in reading probe model ([`2d57102`](https://github.com/datajoint/element-array-ephys/commit/2d57102880d872cf1a4ec037eee5892a87536ff2))

### Unknown

* Merge pull request #205 from MilagrosMarin/chore/update-gha

chore(.github): add new GitHub Action callers ([`d091ffc`](https://github.com/datajoint/element-array-ephys/commit/d091ffc86b6818fdcf16bfdafaa4f8829d771e7b))

* Merge pull request #202 from ttngu207/main

fix(spikeglx): minor bugfix in reading npx probe model of older versions ([`780352b`](https://github.com/datajoint/element-array-ephys/commit/780352b51c3d002d6748bd54f61bd79169e07d95))

## v0.3.5 (2024-08-19)

### Fix

* fix(spikeglx): minor bugfix ([`6764f8c`](https://github.com/datajoint/element-array-ephys/commit/6764f8c1adb9a80569f75233028e551cf58d8917))

* fix: minor bugfix ([`e8870b9`](https://github.com/datajoint/element-array-ephys/commit/e8870b94cf6dc09b251e268c4102fb4b82149da2))

* fix(probe): better handling of different Neuropixels probe types ([`aaec763`](https://github.com/datajoint/element-array-ephys/commit/aaec76339954b17a2dbef8aeaa84e92e64bdad35))

* fix(probe_geometry): bugfix in x_coords for probe with staggered electrode positions ([`54d4fac`](https://github.com/datajoint/element-array-ephys/commit/54d4facd38a79ba9b6e40c01174fdb04e6dee43d))

### Unknown

* Merge pull request #199 from ttngu207/main

fix(probe): better handling of different Neuropixels probe types and SpikeGLX meta loader ([`71d9a42`](https://github.com/datajoint/element-array-ephys/commit/71d9a42b28280d42b021b2e42d492f4918f07cd2))

* update: version and changelog ([`f754392`](https://github.com/datajoint/element-array-ephys/commit/f75439241693f82e75927d23637a4ae471dd6377))

* rearrange: explicitly call `probe.create_neuropixels_probe_types()` to create entries in ProbeType ([`46679e6`](https://github.com/datajoint/element-array-ephys/commit/46679e605e116e13a2cc373148ea24127a2fc447))

* Merge branch &#39;dev_separated_create_probe&#39; into nei_nienborg ([`7f52f59`](https://github.com/datajoint/element-array-ephys/commit/7f52f594ac4d24b2210cc3e2bee4adf0f4c3c913))

* Merge pull request #198 from BrainCOGS/adding-raw-strings-and-package-minimum

Fix regex patterns and add minimum version for scikit-image ([`27c56ea`](https://github.com/datajoint/element-array-ephys/commit/27c56ea92ba0c5d089a2c1e77cbffb52d51dcf6c))

* Added minimum version to the setup.py for scikit-image ([`711dd48`](https://github.com/datajoint/element-array-ephys/commit/711dd48a5396d1e7ba36410c5f141be6940e9c11))

* Provided raw annotation for strings with unsupported escape regex sequences ([`f59b4ab`](https://github.com/datajoint/element-array-ephys/commit/f59b4abf3f1dae42990ef02cd3ff1e6c341aa861))

* Merge pull request #185 from datajoint/pytest

Add pytest ([`9299142`](https://github.com/datajoint/element-array-ephys/commit/9299142605c4b16c14edfe9a44f40f242f25839a))

* apply black formatting ([`333f411`](https://github.com/datajoint/element-array-ephys/commit/333f4118ecbf3eee348fa3671b7da3249302167b))

* update CHANGELOG.md &amp; bump version ([`a3426eb`](https://github.com/datajoint/element-array-ephys/commit/a3426ebe9d9b03f61bef231c17032d5ad2e5c8cd))

* tested version of pytest suite ([`9c033c4`](https://github.com/datajoint/element-array-ephys/commit/9c033c4f355a831dabb8537cfb12dba76c8badab))

* switch ephys to no_curation in tutorial notebook ([`3ae2cbc`](https://github.com/datajoint/element-array-ephys/commit/3ae2cbcc558f17968403e00c983eca1049e51721))

* move tutorial_pipeline.py to tests ([`591a0ed`](https://github.com/datajoint/element-array-ephys/commit/591a0ed0857601517a62acd73967c726269e5eb2))

* setup pytest fixture ([`92026a6`](https://github.com/datajoint/element-array-ephys/commit/92026a614ae61b874d4a4692acc5fc0ad06bd560))

* Merge pull request #183 from MilagrosMarin/main

Minor change: remove pypi release from `release.yaml` ([`cc36465`](https://github.com/datajoint/element-array-ephys/commit/cc36465dc56a9e299a86e54329e806899c6bcf73))

* update version and changelog ([`5cfc269`](https://github.com/datajoint/element-array-ephys/commit/5cfc26921633e89df5fb16637dd61b88361d73d7))

* remove PyPi release ([`dc7863e`](https://github.com/datajoint/element-array-ephys/commit/dc7863edde2431114db8992cb419782b28eaa3ce))

* Merge pull request #182 from datajoint/staging

fix(probe_geometry): bugfix in x_coords for probe with staggered electrode positions ([`95d25f2`](https://github.com/datajoint/element-array-ephys/commit/95d25f2f76e3e8435eb3c8b199437df581aa3916))

* Merge pull request #181 from ttngu207/main

fix(probe_geometry): bugfix in x_coords for probe with staggered electrode positions ([`d65b70f`](https://github.com/datajoint/element-array-ephys/commit/d65b70fd56c89d851b8c29819fe70a219cb81838))

* update(version): update CHANGELOG and version ([`2e79f3d`](https://github.com/datajoint/element-array-ephys/commit/2e79f3d984272ce97709e7c83bb153ab6a2a452a))

## v0.3.1 (2024-01-04)

### Unknown

* Merge pull request #178 from MilagrosMarin/main

Update CHANGELOG and `version.py` ([`6184b2f`](https://github.com/datajoint/element-array-ephys/commit/6184b2fa51db9e430967bbb618b15c0b65549613))

* CHANGELOG and `version.py` updated ([`bfc0c0a`](https://github.com/datajoint/element-array-ephys/commit/bfc0c0a7e31903cf2f74d5675f06a896e4705769))

* Merge pull request #176 from kushalbakshi/main

Update setup dependencies + tutorial setup + fix diagram ([`0174478`](https://github.com/datajoint/element-array-ephys/commit/01744781b27ee4cad16c18bb2e6a1fea175e038c))

* Minor fixes and updates to notebook ([`1ea7c89`](https://github.com/datajoint/element-array-ephys/commit/1ea7c89993465eaa34e02864f654c654b9e7285c))

* Fix typo in setup.py ([`b919ca3`](https://github.com/datajoint/element-array-ephys/commit/b919ca34432c3189d934a3a75bdb071fe8bcb6b9))

* Black formatting ([`16d36d5`](https://github.com/datajoint/element-array-ephys/commit/16d36d56fad9e0e5b97af2fa57df065a664917cf))

* Move dj_config setup to `tutorial_pipeline.py` ([`0dbdde7`](https://github.com/datajoint/element-array-ephys/commit/0dbdde70a27054f5489399daca4743f40c34ce29))

* Remove PyPI versioning in setup ([`b979fec`](https://github.com/datajoint/element-array-ephys/commit/b979feca44468a33601af36c1db3f917993844df))

* Markdown structural edits ([`b2c4901`](https://github.com/datajoint/element-array-ephys/commit/b2c4901095c6d16b43302bbb4da43a0ad813bc6f))

* Run tutorial notebook ([`43ff0d4`](https://github.com/datajoint/element-array-ephys/commit/43ff0d42a1abe89f6bf3f3f334b32773ca0d2c37))

* Minor fixes to README ([`da2239d`](https://github.com/datajoint/element-array-ephys/commit/da2239dbc06dc018f78823c6c36a7f83ff48a5d4))

* Merge branch &#39;main&#39; of https://github.com/kushalbakshi/element-array-ephys ([`9a8865f`](https://github.com/datajoint/element-array-ephys/commit/9a8865f74ec9d4c1e5b661d04bd3f54932ac53e4))

* Update path to flowchart diagram ([`f424a0f`](https://github.com/datajoint/element-array-ephys/commit/f424a0f7726e8dcfafe99b2f053af194d856a536))

* Fix heading styling ([`6323d4e`](https://github.com/datajoint/element-array-ephys/commit/6323d4ee7ac4cd07b494ce93cca8ac159c0bf843))

* Minor updates for formatting and structure ([`8717528`](https://github.com/datajoint/element-array-ephys/commit/87175289eb3a0aa1d401bb41f5b9f7e73c62659a))

* Added diagram_flowchart.svg ([`bf63fe1`](https://github.com/datajoint/element-array-ephys/commit/bf63fe1b6887075a123de99a5b00fd20f9ee9561))

* Added diagram_flowchart.svg ([`4a4104c`](https://github.com/datajoint/element-array-ephys/commit/4a4104c92a370162a4cb51642d337b3567ef28fe))

* Merge branch &#39;datajoint:main&#39; into codespace ([`b66ccee`](https://github.com/datajoint/element-array-ephys/commit/b66cceeda3305c4af75befaaee9e91aa2704bb19))

* Update diagram_flowchart.drawio ([`4f58c68`](https://github.com/datajoint/element-array-ephys/commit/4f58c68562820c583ac04fccd8a394e729adafdb))

* Merge pull request #175 from A-Baji/main

revert: :memo: revert docs dark mode cell text color ([`d24f936`](https://github.com/datajoint/element-array-ephys/commit/d24f936b15cfbb138a4490d3cec3fbd1a5a84e69))

* revert: :memo: revert table style ([`62172c4`](https://github.com/datajoint/element-array-ephys/commit/62172c459c643e66b559c7c4af1943901a890a89))

* Merge pull request #174 from MilagrosMarin/update_tutorial

Improvements in `element-array-ephys` tutorial and README ([`fe4a844`](https://github.com/datajoint/element-array-ephys/commit/fe4a8444ef213fd70625053ec776d6081ac695c6))

* tutorial run with included outputs complete ([`2a59ea0`](https://github.com/datajoint/element-array-ephys/commit/2a59ea0e19ceb11ae43892f7309a119a3ffdfa57))

* Tutorial run with included outputs ([`40eec3e`](https://github.com/datajoint/element-array-ephys/commit/40eec3e8441998970e16ebb34f375f3e6647fd8d))

* Revert deleting `SessionDirectory` insertion ([`ad69298`](https://github.com/datajoint/element-array-ephys/commit/ad692986c3ba73f0885f401e8114bd66e17c4826))

* add markdown in setup ([`abc82ba`](https://github.com/datajoint/element-array-ephys/commit/abc82ba215d67912482981fe2b2766e7a4bccff8))

* ephys tutorial preliminary review to mirror dlc ([`9e16a23`](https://github.com/datajoint/element-array-ephys/commit/9e16a23ff1c8ab7002a013ab4bf4057cd9902253))

* Merge pull request #173 from kushalbakshi/codespace

Add DevContainers + Codespaces tutorial ([`140384e`](https://github.com/datajoint/element-array-ephys/commit/140384ee293a366d03900e490fe03413b7d8531b))

* review PR tutorial ([`733d2b1`](https://github.com/datajoint/element-array-ephys/commit/733d2b1ee9702a6d4e391b6ce62e373534ffdd0a))

* Fix typo in tutorial heading ([`820b282`](https://github.com/datajoint/element-array-ephys/commit/820b282e29eab4793c56ab99ac29ac897f3bdd33))

* Merge branch &#39;codespace&#39; of https://github.com/kushalbakshi/element-array-ephys into codespace ([`a993b8d`](https://github.com/datajoint/element-array-ephys/commit/a993b8d99a964c1cfb599bc7411f1ff27a0c7c9b))

* Updated diagram_flowchart.svg ([`a376c90`](https://github.com/datajoint/element-array-ephys/commit/a376c90a6f495f84af9611b8e8293272892c3c29))

* Fix typo ([`d1657b2`](https://github.com/datajoint/element-array-ephys/commit/d1657b2c57c1d8733cb88bce326511679a575fe3))

* Update README + minor fixes ([`b9fd4a3`](https://github.com/datajoint/element-array-ephys/commit/b9fd4a35f19587ea6527d968bb6b42e5afa880b2))

* Update diagram_flowchart.drawio ([`a08736c`](https://github.com/datajoint/element-array-ephys/commit/a08736cd58b2c43bdeb6e9d0d35b69ce818174f8))

* Update diagram_flowchart.svg ([`f9fc3ec`](https://github.com/datajoint/element-array-ephys/commit/f9fc3ec002160fb0bfe36c579019637bd6ab285e))

* Updated diagram_flowchart.svg ([`bb2f507`](https://github.com/datajoint/element-array-ephys/commit/bb2f507704b7b15bbe0f887dffee0c98018828e3))

* Update diagram_flowchart.drawio ([`6328398`](https://github.com/datajoint/element-array-ephys/commit/632839825214c6c66baa29cd7136bc0bc46f0f3a))

* Complete demo notebooks ([`21fde13`](https://github.com/datajoint/element-array-ephys/commit/21fde1351084c6f73751dd47f0024c9b9e6487ad))

* Black formatting ([`5d57ff2`](https://github.com/datajoint/element-array-ephys/commit/5d57ff2c1d12963391b3318d091a0ac1ca66db6d))

* Update demo presentation notebook ([`1da15db`](https://github.com/datajoint/element-array-ephys/commit/1da15dbd1bc9d759fc2c3cf1dd6e1388f5645fad))

* Add demo notebooks ([`d02d8a5`](https://github.com/datajoint/element-array-ephys/commit/d02d8a585b575ea60b01caae953df4176a40de01))

* Completed tutorial ([`7bf9f9f`](https://github.com/datajoint/element-array-ephys/commit/7bf9f9f3cf80f32963ed421221a4cd405aef6dd8))

* Update root_data_dir in Dockerfile ([`d5430aa`](https://github.com/datajoint/element-array-ephys/commit/d5430aa93b507baf4923acda3d3eb8663e480a23))

* Update Dockerfile and tutorial_pipeline to fix errors ([`1717054`](https://github.com/datajoint/element-array-ephys/commit/1717054c4eefb1176be2a54bc9e75b80508b62a0))

* Use `session_with_datetime` for tutorial ([`ce6e3bf`](https://github.com/datajoint/element-array-ephys/commit/ce6e3bf8ee3968f5b1d1b97e1d7d238272b6c073))

* Update `get_logger` to `dj.logger` ([`b2180c4`](https://github.com/datajoint/element-array-ephys/commit/b2180c457e86303ac816bd0acb94c99fb1097821))

* Markdown improvements in tutorial ([`38c50fb`](https://github.com/datajoint/element-array-ephys/commit/38c50fbaad8cc2a3d3d717ed4c2d5a577fa908e9))

* Upsdate tutorial markdown ([`4190925`](https://github.com/datajoint/element-array-ephys/commit/41909257e34b37c9197943dc75d855c31f9cda89))

* Merge branch &#39;codespace&#39; of https://github.com/kushalbakshi/element-array-ephys into codespace ([`69cef22`](https://github.com/datajoint/element-array-ephys/commit/69cef2204070e258e40e7ef43ba65200be3d560f))

* Update `.gitignore` to include Codespaces ([`f5ab71d`](https://github.com/datajoint/element-array-ephys/commit/f5ab71d8abfcfe973d9792e91307ed705d56f54b))

* Update root data dir env ([`1bea230`](https://github.com/datajoint/element-array-ephys/commit/1bea230d0789be6632c8dbb78139d9a2b8f92421))

* Add tutorial notebook ([`caf5c91`](https://github.com/datajoint/element-array-ephys/commit/caf5c9109d43e373c262d3757c0bc3edd54d416f))

* Allow build step in docker-compose ([`dcd768a`](https://github.com/datajoint/element-array-ephys/commit/dcd768a0e7bc799b9da968ed8831738d1facbee1))

* Black formatting ([`6c6afe4`](https://github.com/datajoint/element-array-ephys/commit/6c6afe4778466b26dbf846c84d1d77daf8672ca7))

* Enable devcontainer builds in CICD ([`5e2d7be`](https://github.com/datajoint/element-array-ephys/commit/5e2d7bef950f70007dc418c9975c22b3488c95a1))

* First commit for codespace compatability ([`5f756c3`](https://github.com/datajoint/element-array-ephys/commit/5f756c36675e5191f2b19d2581dcf1a4b0991729))

* Merge pull request #169 from ttngu207/new_spikeglx_and_probeinterface

New spikeglx and probeinterface ([`3b8efe5`](https://github.com/datajoint/element-array-ephys/commit/3b8efe52fcc16eae13d918a11dc5c1e89378c93e))

* address PR comments

Co-authored-by: Dimitri Yatsenko &lt;dimitri@datajoint.com&gt; ([`f42f1fc`](https://github.com/datajoint/element-array-ephys/commit/f42f1fcff03f0c312d8e09ea50828bf2a77b33b5))

* address PR comments

Co-authored-by: Dimitri Yatsenko &lt;dimitri@datajoint.com&gt; ([`452ff31`](https://github.com/datajoint/element-array-ephys/commit/452ff31c952f641f329771c22c39a4e6845d7588))

* bugfix ([`4407678`](https://github.com/datajoint/element-array-ephys/commit/44076782dcb86af3309fe2bda909d971f9819266))

* bugfix in geomap to shank conversion ([`1514613`](https://github.com/datajoint/element-array-ephys/commit/1514613fb9b74c5be628f2d5e53882ea6f4e7da1))

* transform geom map to shank map ([`9857aef`](https://github.com/datajoint/element-array-ephys/commit/9857aef544ebabc84e4906b421420bc2407b55a6))

* update spikeglx loader to handle spikeglx ver 20230202 ([`3994fc7`](https://github.com/datajoint/element-array-ephys/commit/3994fc75b10f6d5a92e6c7b664067641716d518e))

* incorporate probeinterface and probe geometry for all npx probes ([`224b1c7`](https://github.com/datajoint/element-array-ephys/commit/224b1c7049c9e246df22fe3a46fbd357d3096d8b))

## v0.2.11 (2023-06-30)

### Unknown

* Merge pull request #165 from kabilar/patch

Patch issue with nullable attributes ([`e4dd98a`](https://github.com/datajoint/element-array-ephys/commit/e4dd98a3541271e041538daa053f158c1b9f8c63))

* Temporarily remove Docker image builds ([`48a1e76`](https://github.com/datajoint/element-array-ephys/commit/48a1e768ad6b8cf05bf519cdcbf0e503aa73e613))

* Format with black ([`d5de596`](https://github.com/datajoint/element-array-ephys/commit/d5de59661c21992dbc9104f6ce8ca9c26e64cc91))

* Update image path ([`2557877`](https://github.com/datajoint/element-array-ephys/commit/25578773db6a478c54486ebcfc9010d7c23fa87e))

* Update text ([`8a764e8`](https://github.com/datajoint/element-array-ephys/commit/8a764e85f0645fb38d2d74f87fdfb73260bb2524))

* Update readme ([`c530671`](https://github.com/datajoint/element-array-ephys/commit/c5306715508891428e69203172091664c6d34c7a))

* Update changelog ([`d1cf13f`](https://github.com/datajoint/element-array-ephys/commit/d1cf13f8595c0fe6d90a0dd029a65b73d8ecec4a))

* Update GitHub Actions ([`71bb8e2`](https://github.com/datajoint/element-array-ephys/commit/71bb8e2a489d044a01c4328027630c5f8f34b6cf))

* Update version and changelog ([`d4f7fe0`](https://github.com/datajoint/element-array-ephys/commit/d4f7fe080eb5fe94f518c2db5b17ffef4448dee2))

* Add default value ([`01ad1e8`](https://github.com/datajoint/element-array-ephys/commit/01ad1e85a3e2db48c198b2d0d32096152ffba295))

* Merge pull request #142 from ttngu207/main

Update kilosort_triggering.py ([`1d30cb8`](https://github.com/datajoint/element-array-ephys/commit/1d30cb81c258d396aba16a38ed20fbfb0e55a052))

* update CHANGELOG ([`5e1f055`](https://github.com/datajoint/element-array-ephys/commit/5e1f0555349a11b5b51498d13e209a28781c1b11))

* Merge branch &#39;datajoint:main&#39; into main ([`c5f20b0`](https://github.com/datajoint/element-array-ephys/commit/c5f20b0063363e4946ee059838db34c0dc3c57ac))

## v0.2.10 (2023-05-26)

### Unknown

* Merge pull request #151 from kabilar/main

Fix readability of tables in dark mode ([`47dea95`](https://github.com/datajoint/element-array-ephys/commit/47dea95466cb771f19807c8e0499efc5e3f2f577))

* Update citation ([`100913e`](https://github.com/datajoint/element-array-ephys/commit/100913e772c64bc482fde088842e184972ba479f))

* Update changelog ([`0bfca62`](https://github.com/datajoint/element-array-ephys/commit/0bfca629a963196470f4dc291ba1678b58a2829c))

* Update CSS ([`15e9ddb`](https://github.com/datajoint/element-array-ephys/commit/15e9ddb4e8fe21f078b623c5eb64131f6255d82a))

* Merge pull request #150 from kabilar/main

Add Kilosort, NWB, and DANDI citations ([`ad9588f`](https://github.com/datajoint/element-array-ephys/commit/ad9588fb1c1293d1b8e598f573b280f06ba4e750))

* Add NWB and DANDI citations ([`af81ef9`](https://github.com/datajoint/element-array-ephys/commit/af81ef973859a0eea890d3c6ff513640a9255e16))

* Update citation page ([`aee35f7`](https://github.com/datajoint/element-array-ephys/commit/aee35f7918d4d9c66fa97d8e1795b66f43e58996))

* Update changelog ([`0ca91fa`](https://github.com/datajoint/element-array-ephys/commit/0ca91fa11ebf11b7dcbe600c628f63fc8c32078c))

* Update changelog ([`f89eae4`](https://github.com/datajoint/element-array-ephys/commit/f89eae42128c2ca3f068aecf05d04ca214f40282))

* Add plugin ([`4436b05`](https://github.com/datajoint/element-array-ephys/commit/4436b056d3408af20d448068f5ed6d27b8486465))

* Remove redirects ([`c798564`](https://github.com/datajoint/element-array-ephys/commit/c798564673444260e2a1094bf2cfd615203283da))

* Update changelog ([`b63031c`](https://github.com/datajoint/element-array-ephys/commit/b63031cf788fe075b410cc23f15d8efc90819895))

* Add citation ([`69e76dd`](https://github.com/datajoint/element-array-ephys/commit/69e76dd58e9785f3d4f93e78c3c7b86006e1eae4))

* Merge pull request #149 from kushalbakshi/main

Fix notebook output in dark mode ([`97a57b1`](https://github.com/datajoint/element-array-ephys/commit/97a57b158b116cf8d0b6e84ecc1fe737f3176366))

## v0.2.9 (2023-05-11)

### Unknown

* Merge branch &#39;main&#39; of https://github.com/kushalbakshi/element-array-ephys ([`e4809ba`](https://github.com/datajoint/element-array-ephys/commit/e4809ba249f3885ff578123eba6e479ad672a9f0))

* Merge pull request #148 from kushalbakshi/main

Fix docs tutorials in dark mode ([`96d1187`](https://github.com/datajoint/element-array-ephys/commit/96d118777d5a0f736ea2ca224eb90526d7637616))

* Dark mode notebooks fix ([`9aab33d`](https://github.com/datajoint/element-array-ephys/commit/9aab33da951bcddb8515f91ade88aec783631113))

## v0.2.8 (2023-04-28)

### Unknown

* Fix docs tutorials in dark mode ([`d2367ce`](https://github.com/datajoint/element-array-ephys/commit/d2367ce4e430273ada830d9c6f4eefdde66c2637))

* Merge pull request #146 from JaerongA/metrics

Remap `metrics.csv` column names ([`6aef807`](https://github.com/datajoint/element-array-ephys/commit/6aef8074a16af945c4c6d928bf480daa6e4d1401))

## v0.2.7 (2023-04-19)

### Unknown

* update changelog and version ([`6b069e6`](https://github.com/datajoint/element-array-ephys/commit/6b069e68efbe933c1324a7afbd10276732c3b49e))

* add column name mapping for metrics.csv ([`c97d509`](https://github.com/datajoint/element-array-ephys/commit/c97d5090c445ed6c4f8595cf64dfb59eb965545e))

## v0.2.6 (2023-04-18)

### Unknown

* Merge pull request #143 from kabilar/main

Update version and changelog for release ([`5abecc3`](https://github.com/datajoint/element-array-ephys/commit/5abecc3ccf0a3e72e5807e67a1a11d875953450e))

* Update `ephys_precluster` ([`5993d6e`](https://github.com/datajoint/element-array-ephys/commit/5993d6eeb1e3ab57d1cca96a47e32440170c0477))

* Update version and changelog ([`e9b66af`](https://github.com/datajoint/element-array-ephys/commit/e9b66aff50f78715420518ec5470ec1e1435abaf))

* Merge `main` of datajoint/element-array-ephys ([`40b5a6d`](https://github.com/datajoint/element-array-ephys/commit/40b5a6df8c884ee36615d3b20fd2f838ac405062))

* Merge pull request #144 from JaerongA/main ([`e487f3a`](https://github.com/datajoint/element-array-ephys/commit/e487f3a2083f3c25bbe160eba1ac59d4707e3793))

* lowercase all column names in metrics.csv ([`f35ba0b`](https://github.com/datajoint/element-array-ephys/commit/f35ba0b383efb8d59448eb6220a6a4dab153f41d))

* Merge pull request #138 from JaerongA/main

Update docs for quality metrics ([`aabc454`](https://github.com/datajoint/element-array-ephys/commit/aabc45420eaead26966309ecedbfec513e89a771))

## v0.2.5 (2023-04-13)

### Unknown

* remove schema tag in SkullReference ([`7192958`](https://github.com/datajoint/element-array-ephys/commit/7192958dcb188f6f72c363eefa333786acfd0216))

* add a new tag ([`b8ef2d9`](https://github.com/datajoint/element-array-ephys/commit/b8ef2d9d2069cce2e0641ea63dd461b60634a39b))

* update schema diagrams to show SkullReference ([`8ffe6df`](https://github.com/datajoint/element-array-ephys/commit/8ffe6dfb9dbe9ee1b421cb05472daf07c1a1428e))

* Update CHANGELOG.md

Co-authored-by: Kabilar Gunalan &lt;kabilar@datajoint.com&gt; ([`166c00a`](https://github.com/datajoint/element-array-ephys/commit/166c00a20a48805cbc3fe8c3dc300b7c9bd8a7ae))

* Update docs/src/concepts.md

Co-authored-by: Kabilar Gunalan &lt;kabilar@datajoint.com&gt; ([`4a304d8`](https://github.com/datajoint/element-array-ephys/commit/4a304d88d77d0f7395916881c0110019ce72d803))

* bump version ([`7ceae8b`](https://github.com/datajoint/element-array-ephys/commit/7ceae8bac60409d50fd105c9ccd13177ff5f4339))

* add schema diagrams ([`eb5b0b1`](https://github.com/datajoint/element-array-ephys/commit/eb5b0b10b7efc98c2aa47ebd8c709758bfa6bfea))

* add quality_metrics.ipynb to mkdocs ([`977b90a`](https://github.com/datajoint/element-array-ephys/commit/977b90a80dec5034aa78eb90f9810df8b0ff942b))

* Update docs/src/concepts.md

Co-authored-by: Kabilar Gunalan &lt;kabilar@datajoint.com&gt; ([`27d5742`](https://github.com/datajoint/element-array-ephys/commit/27d57420e6ce7107c1b3b73acb5ac436c0155a4a))

* Update element_array_ephys/ephys_report.py

Co-authored-by: Kabilar Gunalan &lt;kabilar@datajoint.com&gt; ([`1e37791`](https://github.com/datajoint/element-array-ephys/commit/1e3779182767b699faac78e952a1e4e64f4e2854))

* Update requirements.txt

Co-authored-by: Kabilar Gunalan &lt;kabilar@datajoint.com&gt; ([`46b911b`](https://github.com/datajoint/element-array-ephys/commit/46b911b6df8a49551e5042b5d422135b4585ffc7))

* add nbformat to requirements.txt ([`2f38cfa`](https://github.com/datajoint/element-array-ephys/commit/2f38cfa099ab62561c6dbbfdbd6e5ea233de233d))

* update requirements.txt ([`eb0d795`](https://github.com/datajoint/element-array-ephys/commit/eb0d7955f53879c511439b77d1f5561a36049e85))

* update concepts.md ([`d362792`](https://github.com/datajoint/element-array-ephys/commit/d362792891e9e4c1e3b5f604d070d32cb8e24a23))

* fix docstring error in qc.py ([`ea03674`](https://github.com/datajoint/element-array-ephys/commit/ea036742dc28421bad116e897055b867eff6c54a))

* add docstring for qc metric tables ([`7a89a7f`](https://github.com/datajoint/element-array-ephys/commit/7a89a7fc5819fa205bf332155f684d9b94529133))

* fix docstring error ([`9cc4545`](https://github.com/datajoint/element-array-ephys/commit/9cc4545681ec3da7d0a1b1230c06e12fbdd2e409))

* Fix typo ([`e0919ae`](https://github.com/datajoint/element-array-ephys/commit/e0919ae14670b1cce1341d216afe08c990fa1285))

* Update docs configuration ([`2dcca99`](https://github.com/datajoint/element-array-ephys/commit/2dcca99e0ed3d4e1d7ea14b44686f8d6a7f0bd5e))

* Fix for `PT_ratio` to `pt_ratio` ([`e4358a5`](https://github.com/datajoint/element-array-ephys/commit/e4358a50c152d940c663b691b74342570f518c30))

* Update kilosort_triggering.py ([`8035648`](https://github.com/datajoint/element-array-ephys/commit/8035648551e6545cd66f2f4d5911241226430302))

## v0.2.4 (2023-03-10)

### Unknown

* Merge pull request #137 from kabilar/main

Update requirements for plotting widget ([`2fa46bd`](https://github.com/datajoint/element-array-ephys/commit/2fa46bd690c6bf4a22494dc2086c506f5ba48bf8))

* Update changelog ([`2eb359f`](https://github.com/datajoint/element-array-ephys/commit/2eb359f354b55235ae350dc89cf4f918449764c2))

* Update changelog ([`ac581b9`](https://github.com/datajoint/element-array-ephys/commit/ac581b9eafdb2848aecb2691509038ed9a9c9c13))

* Add dependency ([`00f4f6d`](https://github.com/datajoint/element-array-ephys/commit/00f4f6d9693e2c470b46e55341ee54f5c2f36df2))

* Add ipywidgets as dependency ([`6840069`](https://github.com/datajoint/element-array-ephys/commit/684006910b979b70fa7c9d9a5614a570da1159d2))

## v0.2.3 (2023-02-14)

### Fix

* fix: :bug: import from __future__ module

this supports backward compability with typing ([`4d9ab28`](https://github.com/datajoint/element-array-ephys/commit/4d9ab28609b161be986cda2872778d7a403277f3))

### Unknown

* Merge pull request #132 from CBroz1/main

Add 0.2.3 release date ([`e39f9d3`](https://github.com/datajoint/element-array-ephys/commit/e39f9d30e23052f4d1bbf85bc7ca459d80e641c2))

* Add 0.2.3 release date ([`eaeef30`](https://github.com/datajoint/element-array-ephys/commit/eaeef306827799448050f0e96fa7d2c03ece750b))

* Merge pull request #125 from CBroz1/main

Adjusting pyopenephys requirement for pypi publication ([`c7f92af`](https://github.com/datajoint/element-array-ephys/commit/c7f92af01da6e975bb25e63a472825e8491f1057))

* New pyopenephys version ([`f07bd44`](https://github.com/datajoint/element-array-ephys/commit/f07bd44abcaa4ac88cded20168b4e8f63b451563))

* Merge branch &#39;main&#39; of https://github.com/datajoint/element-array-ephys ([`905588f`](https://github.com/datajoint/element-array-ephys/commit/905588f5a3f03ae813a4ff25e202aa69b9e47ec7))

* Merge pull request #128 from ttngu207/main

fix removing outdated files/directory ([`53f0312`](https://github.com/datajoint/element-array-ephys/commit/53f0312272ffbf7989c40b2ba91c9e7a33a81faa))

* Update kilosort_triggering.py ([`a34e437`](https://github.com/datajoint/element-array-ephys/commit/a34e437380868498975735e97561bb406a0cc70f))

* fix removing outdated files/directory ([`6fb65b3`](https://github.com/datajoint/element-array-ephys/commit/6fb65b376c25bd35dd4e665519c20e3ce8f33f4d))

* minor improvement ([`7c6dc37`](https://github.com/datajoint/element-array-ephys/commit/7c6dc374184a90e3211cef63bf0a4c31dd7a35fc))

* Merge pull request #126 from sidhulyalkar/main

Fix multiple hash kilosort output issue ([`b619bd0`](https://github.com/datajoint/element-array-ephys/commit/b619bd05d77246808d61627eacaa4e6cad4aa69a))

* edit comment ([`f17f299`](https://github.com/datajoint/element-array-ephys/commit/f17f299f8a379b2e6de67c27c51e518d692f0f56))

* Fix issue where hash is changed(different paramset) and trying to rerun processing ([`7774492`](https://github.com/datajoint/element-array-ephys/commit/7774492e21fd361560393e5d2bde906adb256e37))

* Merge branch &#39;main&#39; of https://github.com/sidhulyalkar/element-array-ephys ([`152cc58`](https://github.com/datajoint/element-array-ephys/commit/152cc586e294de185aa799e50af9717b4b6948bf))

* Merge branch &#39;main&#39; of https://github.com/sidhulyalkar/element-array-ephys ([`2df6280`](https://github.com/datajoint/element-array-ephys/commit/2df6280b09c36ace534b11726220140afc6d2431))

* Merge branch &#39;main&#39; of https://github.com/sidhulyalkar/element-array-ephys ([`06568f4`](https://github.com/datajoint/element-array-ephys/commit/06568f445a8bc8da1b0c34eb457899626de75dba))

* Merge branch &#39;main&#39; of https://github.com/sidhulyalkar/element-array-ephys ([`e7f6060`](https://github.com/datajoint/element-array-ephys/commit/e7f6060467f02510558e7b86e9e664c7cbdbc38d))

* Merge branch &#39;run_kilosort&#39; of https://github.com/sidhulyalkar/element-array-ephys ([`4e195c3`](https://github.com/datajoint/element-array-ephys/commit/4e195c3b173cfdb94371311c1d6be6babad7b75c))

* Merge branch &#39;main&#39; of https://github.com/sidhulyalkar/element-array-ephys ([`f5ca7e8`](https://github.com/datajoint/element-array-ephys/commit/f5ca7e87ad8bc51c0801c7b2504e0fb8092a3a08))

* Added Code of Conduct ([`195c61e`](https://github.com/datajoint/element-array-ephys/commit/195c61e8e825d90701b84b1c18fa204fa56c8bc3))

* Simplify import ([`47f6a07`](https://github.com/datajoint/element-array-ephys/commit/47f6a07ad030dff9997272a4f74abe7962593277))

* pyopenephys import workaround ([`2a742e6`](https://github.com/datajoint/element-array-ephys/commit/2a742e694326d2937a07587469730110f1f11b39))

* Merge pull request #124 from CBroz1/main

Cleanup docstrings, add notebook render ([`d5b9586`](https://github.com/datajoint/element-array-ephys/commit/d5b95860977485e1020500d72f2bf18576a18aad))

* Apply suggestions from code review

Co-authored-by: JaerongA &lt;jaerong.ahn@datajoint.com&gt; ([`4796056`](https://github.com/datajoint/element-array-ephys/commit/4796056c1bdb98039cc52e43a4491dc43f6bcfef))

* Merge pull request #3 from JaerongA/main

fix: :bug: import from __future__ to support backward compatibility ([`fd94939`](https://github.com/datajoint/element-array-ephys/commit/fd94939eb518e8727706f3dff56ebc35bb3fcb5f))

* Merge branch &#39;main&#39; into main ([`084ada2`](https://github.com/datajoint/element-array-ephys/commit/084ada258f9935f6d3636fd31c1a962b3be0a9aa))

* Adjust dependencies 2 ([`a28cf13`](https://github.com/datajoint/element-array-ephys/commit/a28cf13118f15fbec515171852027965b8c433ad))

* Adjust dependencies ([`45f846c`](https://github.com/datajoint/element-array-ephys/commit/45f846cf75004b9e5ea6e6580f6f16209515c6f0))

* Fix typing bug ([`888e7f7`](https://github.com/datajoint/element-array-ephys/commit/888e7f743fcfc10dc190c3020523aeb1547c8380))

* interface requirement to pip installable ([`9ff2e04`](https://github.com/datajoint/element-array-ephys/commit/9ff2e04cb879fe8aae0fe985a661f7fe8761f79b))

* add extras_require nwb install option to docs ([`8045879`](https://github.com/datajoint/element-array-ephys/commit/80458796ebda66e81312211677a0f4aa93100295))

* Version bump, changelog ([`9d03350`](https://github.com/datajoint/element-array-ephys/commit/9d0335047fe67e02ebf5ccc6da395d5bdbeda3df))

* Add extras_require for dev and nwb ([`e01683c`](https://github.com/datajoint/element-array-ephys/commit/e01683ca2241e457ccfa6f4d61914808400d663e))

* Spelling ([`56eb68a`](https://github.com/datajoint/element-array-ephys/commit/56eb68a96f1552aaeed967d4c46ccad45c8eabcd))

* Fix docstrings ([`0980242`](https://github.com/datajoint/element-array-ephys/commit/0980242d1502d09612b424ce7b9f06a250d11342))

* Add tutorial notebook renders ([`96bb6fa`](https://github.com/datajoint/element-array-ephys/commit/96bb6fa10207782e2d5249eda76007ebc453567d))

* More spellcheck; Markdown linting ([`64e7dc6`](https://github.com/datajoint/element-array-ephys/commit/64e7dc690c7cbf1f4051d4cbd3e5d29f3bfd9218))

* Update License 2023 ([`4ef0b6d`](https://github.com/datajoint/element-array-ephys/commit/4ef0b6db9a27fd7ee68fcc48744ee98981947156))

* Spellcheck pass ([`ea980e9`](https://github.com/datajoint/element-array-ephys/commit/ea980e9d35a7582ff1953651812e61a736931c9d))

* Remove unused import ([`b3c0786`](https://github.com/datajoint/element-array-ephys/commit/b3c0786329d6c60c00e092aa0f3a10e920970c20))

* blackify ([`0e5a1c6`](https://github.com/datajoint/element-array-ephys/commit/0e5a1c64d8e7913cdb70ef701c5e53a83225fbcf))

* Merge branch &#39;main&#39; of https://github.com/datajoint/element-array-ephys ([`b250f2d`](https://github.com/datajoint/element-array-ephys/commit/b250f2dd15395876196029789667250cf331a6ee))

* Merge branch &#39;main&#39; of https://github.com/JaerongA/element-array-ephys ([`147550c`](https://github.com/datajoint/element-array-ephys/commit/147550c6fee8a6d510d877be627025f0c710aba8))

* Merge branch &#39;main&#39; of https://github.com/datajoint/element-array-ephys ([`06db0f8`](https://github.com/datajoint/element-array-ephys/commit/06db0f84f2b5a686a3fe87ef14d3be196e7861b3))

* Update element_array_ephys/export/nwb/nwb.py

Co-authored-by: Kabilar Gunalan &lt;kabilar@datajoint.com&gt; ([`0ec5e3d`](https://github.com/datajoint/element-array-ephys/commit/0ec5e3d135f62013083ce2799db927e23ad1d10e))

* Add nwb.py from run_kilosort branch ([`31e46cd`](https://github.com/datajoint/element-array-ephys/commit/31e46cd30f0467f43aca18ca450ec72667656417))

* Merge branch &#39;main&#39; of https://github.com/datajoint/element-array-ephys ([`5711a12`](https://github.com/datajoint/element-array-ephys/commit/5711a12ac2f2ed90abf8a27b9120c017c339090a))

* Merge branch &#39;main&#39; of https://github.com/datajoint/element-array-ephys ([`a8e93d1`](https://github.com/datajoint/element-array-ephys/commit/a8e93d162b1d70f0eb704973141198f750e52f76))

## v0.2.2 (2023-01-11)

### Unknown

* Merge pull request #122 from JaerongA/main

Revert import order ([`eababbc`](https://github.com/datajoint/element-array-ephys/commit/eababbc4bb02ecc137641a19163dea8a5c8b6785))

* add back deleted version tags ([`bd0e76a`](https://github.com/datajoint/element-array-ephys/commit/bd0e76a685cb143bc631bc03a64df45123d65138))

* merge upstream &amp; resolve conflicts ([`1feff92`](https://github.com/datajoint/element-array-ephys/commit/1feff92b61fb473a2c1464ed130fa2fa1b7f58df))

* Merge pull request #120 from kushalbakshi/main

Docstring changes for docs API ([`623a381`](https://github.com/datajoint/element-array-ephys/commit/623a38112e6d7404421eece243de1d385faeb663))

* Merge branch &#39;main&#39; of https://github.com/datajoint/element-array-ephys ([`523f2e9`](https://github.com/datajoint/element-array-ephys/commit/523f2e97d3245f60c49f75496beae76d92bc074d))

* Update CHANGELOG to resolve merge conflicts ([`e613fc4`](https://github.com/datajoint/element-array-ephys/commit/e613fc45d34c83726d4bf032cb61c402a574c065))

* Docstring changes for docs API ([`87a7849`](https://github.com/datajoint/element-array-ephys/commit/87a7849a021ec4076624836a2b5d288448ada909))

* update CHANGELOG.md ([`5c7a772`](https://github.com/datajoint/element-array-ephys/commit/5c7a7722b46196a25e3cb85603a3487113dd6592))

* revert import order in __init__.py ([`956b96e`](https://github.com/datajoint/element-array-ephys/commit/956b96ec563b17912efe14d0764d6fedfe49f8a5))

* Add E402 in pre-commit-config ([`f4f283a`](https://github.com/datajoint/element-array-ephys/commit/f4f283a4209492e55dc82648434ecba55e05acfd))

## v0.2.1 (2023-01-09)

### Chore

* chore: :loud_sound: update CHANGELOG ([`2d43321`](https://github.com/datajoint/element-array-ephys/commit/2d4332189655eeffff7fd41fa071adcc2754ff16))

* chore: :rewind: revert formatting in concepts.md ([`c16b6bd`](https://github.com/datajoint/element-array-ephys/commit/c16b6bdcbadb97fae2b0c21d3dc8c13308fd4131))

* chore(deps): :pushpin: unpin plotly ([`8504b97`](https://github.com/datajoint/element-array-ephys/commit/8504b9724a129c617a8264cbbeb1c26c8a696d8e))

### Documentation

* docs: :memo: add | update docstrings ([`4999d64`](https://github.com/datajoint/element-array-ephys/commit/4999d64980e4cf278f872159c7d327387939ee12))

* docs: :memo: name change &amp; add docstring ([`d9c75c8`](https://github.com/datajoint/element-array-ephys/commit/d9c75c8ea425eb38dc8e714639ad341edf39cafd))

### Refactor

* refactor: :pencil2: fix typos ([`efca82e`](https://github.com/datajoint/element-array-ephys/commit/efca82e352adee21c6979e94078c1c82b8b423aa))

* refactor(deps): :heavy_minus_sign: remove ibllib deps and add acorr func ([`c613164`](https://github.com/datajoint/element-array-ephys/commit/c613164ae90cac220c1726a9eb2e12f336f876db))

### Unknown

* Merge pull request #116 from JaerongA/ephys_test

modify build_electrodes function ([`0f518f1`](https://github.com/datajoint/element-array-ephys/commit/0f518f1b0cd60ffeb22a6345cadfbe0e72ecf2b3))

* Update element_array_ephys/probe.py

Co-authored-by: Chris Brozdowski &lt;CBrozdowski@yahoo.com&gt; ([`59c72c4`](https://github.com/datajoint/element-array-ephys/commit/59c72c4670abe6f0649eb829af14b289d605b98a))

* Update element_array_ephys/probe.py

Co-authored-by: Chris Brozdowski &lt;CBrozdowski@yahoo.com&gt; ([`c983fa8`](https://github.com/datajoint/element-array-ephys/commit/c983fa8967e5c0a20b2a7f7851d5622377ccb16e))

* Update element_array_ephys/probe.py

Co-authored-by: Chris Brozdowski &lt;CBrozdowski@yahoo.com&gt; ([`8e53f54`](https://github.com/datajoint/element-array-ephys/commit/8e53f5448b37dee63b678e3f641dcf505149c7ac))

* remove redundant type hinting ([`19c447a`](https://github.com/datajoint/element-array-ephys/commit/19c447af3ba77f8330ac7ace59e4e0e9e49fde52))

* Merge branch &#39;main&#39; of https://github.com/datajoint/element-array-ephys into ephys_test ([`52a06e9`](https://github.com/datajoint/element-array-ephys/commit/52a06e93b7591a01d7ead857eb96ae53dbbb8cfc))

* Merge pull request #118 from JaerongA/main

Add pre-commit &amp; update docstrings &amp; various fixes ([`61cb23a`](https://github.com/datajoint/element-array-ephys/commit/61cb23a33be65df8d06d2e344548be8d160f25d2))

* pull upstream &amp; resolve merge conflicts ([`26b6be9`](https://github.com/datajoint/element-array-ephys/commit/26b6be9788af03fbba53adb31447623d21ee43da))

* Merge branch &#39;datajoint:main&#39; into main ([`c3ad36f`](https://github.com/datajoint/element-array-ephys/commit/c3ad36f8dfacfb3d58beaaee065c5d75d0a54b28))

* Merge pull request #121 from ttngu207/main

parameterize run_CatGT step from parameters retrieved from `ClusteringParamSet` table ([`24df134`](https://github.com/datajoint/element-array-ephys/commit/24df134629819c17601eec7addf2ae4f359cc567))

* Update CHANGELOG.md ([`f5dff5c`](https://github.com/datajoint/element-array-ephys/commit/f5dff5cf04feae0d4c3c142fcc8769bb86d9c0a6))

* catGT checks and parameterizable ([`0ade344`](https://github.com/datajoint/element-array-ephys/commit/0ade344dc792ffce2dc25d412ea97a45b135c4d8))

* improve validate_file logic ([`63dbd12`](https://github.com/datajoint/element-array-ephys/commit/63dbd12011b1c9978da41664d83f4b36d3a42a19))

* update CHANGELOG.md ([`294d4f5`](https://github.com/datajoint/element-array-ephys/commit/294d4f5b8063f32bdc2f82858f10e7b8a0804e0d))

* Update element_array_ephys/version.py

Co-authored-by: Chris Brozdowski &lt;CBrozdowski@yahoo.com&gt; ([`b343c15`](https://github.com/datajoint/element-array-ephys/commit/b343c15b20919658b36e3ea0ddb3eef3f82dbf02))

* Apply suggestions from code review

Co-authored-by: Kabilar Gunalan &lt;kabilar@datajoint.com&gt; ([`4651707`](https://github.com/datajoint/element-array-ephys/commit/4651707843f3d78448563bbecc982592b99da035))

* Apply suggestions from code review

Co-authored-by: Kabilar Gunalan &lt;kabilar@datajoint.com&gt; ([`8920b00`](https://github.com/datajoint/element-array-ephys/commit/8920b00b617b2fc1932e0824740f5bd168715d47))

* Update CHANGELOG.md

Co-authored-by: Kabilar Gunalan &lt;kabilar@datajoint.com&gt; ([`b29e161`](https://github.com/datajoint/element-array-ephys/commit/b29e1613dd25280ab8e6745d2f9110e96287ec9f))

* Update CHANGELOG.md

Co-authored-by: Kabilar Gunalan &lt;kabilar@datajoint.com&gt; ([`2889293`](https://github.com/datajoint/element-array-ephys/commit/28892939ba2b74aaa748ce4458e31f29e74f9f2a))

* Apply suggestions from code review

Co-authored-by: Kabilar Gunalan &lt;kabilar@datajoint.com&gt;
Co-authored-by: Chris Brozdowski &lt;CBrozdowski@yahoo.com&gt; ([`7e213c5`](https://github.com/datajoint/element-array-ephys/commit/7e213c5472b8830a0dedc7f2a727470e02bbfa51))

* Update setup.py

Co-authored-by: Chris Brozdowski &lt;CBrozdowski@yahoo.com&gt; ([`5649b61`](https://github.com/datajoint/element-array-ephys/commit/5649b61457ef8c74f1e3f3938870edeceea158c3))

* ✏️ fix typos ([`dc85370`](https://github.com/datajoint/element-array-ephys/commit/dc853709848062dc146fa21bf5fb1d11a81a4022))

* pull from upstream ([`bda30a3`](https://github.com/datajoint/element-array-ephys/commit/bda30a34214ccf47ac0ecd6fb0ab0bc9ff6101c1))

* Merge pull request #119 from CBroz1/qc

Add QC dashboard ([`08d1291`](https://github.com/datajoint/element-array-ephys/commit/08d12914253cc7b4da1170db9dc233ba49cbc283))

* Remove interface 42 dependence ([`4f6c301`](https://github.com/datajoint/element-array-ephys/commit/4f6c30199ee2cec36dd86b5ddeb27062955f1f90))

* Cleanup ([`eed1eee`](https://github.com/datajoint/element-array-ephys/commit/eed1eeeb3dd23a33764bdde97b8e0f32ae245a8d))

* Cleanup, add comments ([`c31c53d`](https://github.com/datajoint/element-array-ephys/commit/c31c53da0f5cfe4a24221bb5cf73886915ff2ffc))

* Version bump + changelog ([`9eb34cd`](https://github.com/datajoint/element-array-ephys/commit/9eb34cdca25f271896c2e2c508db1cfb33d82373))

* Remove unused import ([`9989dc7`](https://github.com/datajoint/element-array-ephys/commit/9989dc741b8808e284a7eab3aa9c292cb1e1966b))

* Revert removed docstring item ([`ddeabc7`](https://github.com/datajoint/element-array-ephys/commit/ddeabc7b9d7ed11ade5ed0249ee5c2b4dc318e41))

* Run isort on qc.py ([`a65cbe4`](https://github.com/datajoint/element-array-ephys/commit/a65cbe49ed565c041f3f744c2d1b8eff2f6843a7))

* WIP: QC dashboard 2 ([`f997fd6`](https://github.com/datajoint/element-array-ephys/commit/f997fd6c82bddd33f6007dc3bea6ced29ed49bb2))

* WIP: QC dashboard ([`c873acf`](https://github.com/datajoint/element-array-ephys/commit/c873acfe9fc9a91eca2f4e6f45bd5c49a26380d6))

* update docstring ([`8561326`](https://github.com/datajoint/element-array-ephys/commit/8561326c3b2d2c97d5eb0a3f28dcd1d5647d4d2f))

* Merge pull request #3 from CBroz1/ja

linter recommended changes, reduce linter exceptions ([`9a78c15`](https://github.com/datajoint/element-array-ephys/commit/9a78c15a5d553e18c19d1038b552a47ae399330b))

* Apply isort ([`904ccb9`](https://github.com/datajoint/element-array-ephys/commit/904ccb9ce2454276e005935bfee2d9b9cf0b181f))

* Apply hooks except isort ([`5645ebc`](https://github.com/datajoint/element-array-ephys/commit/5645ebc15cd29c005df36ee17cd316824ff50159))

* See details

- Apply black
- Remove trailing whitespace
- Reduce flake8 exceptions
- Move imports to top
- Remove unused imports (e.g., re, pathlib, log_from_json)
- Add __all__ to init files specifying imports loaded
- Add `noqa: C901` selectively where complexity is high
- l -&gt; line for _read_meta func in spikeglx.py
- give version a default value before loading ([`181677e`](https://github.com/datajoint/element-array-ephys/commit/181677e7622e1102679131974cf0fc567604ede1))

* fix docstrings ([`47de1d5`](https://github.com/datajoint/element-array-ephys/commit/47de1d5eac6b24608b20f74c9551b6c8040c9bd3))

* update concepts.md ([`bc95946`](https://github.com/datajoint/element-array-ephys/commit/bc95946656ebd5100b2e3c584b18b857451b3ab8))

* update version.py ([`6a997e1`](https://github.com/datajoint/element-array-ephys/commit/6a997e17aa56298b19d71c915a7116320b5d7ad1))

* pre-commit ignores docs, .github ([`e52c12e`](https://github.com/datajoint/element-array-ephys/commit/e52c12ed3509ce145853e709334043b8bd1c3272))

* update changelog.md ([`8f4ca3f`](https://github.com/datajoint/element-array-ephys/commit/8f4ca3fc216690a526392617cbc65a634da5f63a))

* add to ignored flake8 rules ([`16de049`](https://github.com/datajoint/element-array-ephys/commit/16de049e86b417895dd38f686a972b4af6df885e))

* import DataJointError ([`434e16a`](https://github.com/datajoint/element-array-ephys/commit/434e16a1013b48f057b0f19b7ee51fd63ac0cc36))

* update docstring ([`12098ce`](https://github.com/datajoint/element-array-ephys/commit/12098ceb5bb1b004998ff3e59d89a4e5c3b17c80))

* add requirements_dev.txt ([`d19f97f`](https://github.com/datajoint/element-array-ephys/commit/d19f97f22a15d609534796ae04257011fe518d35))

* add pre-commit-config ([`278e2f2`](https://github.com/datajoint/element-array-ephys/commit/278e2f2aeb462b3313abcfbaf93de2e206332fcb))

* Update element_array_ephys/ephys_report.py

Co-authored-by: Chris Brozdowski &lt;CBrozdowski@yahoo.com&gt; ([`b9fb872`](https://github.com/datajoint/element-array-ephys/commit/b9fb872621449710be8409239c1c0379cc59ed0c))

* Update element_array_ephys/ephys_report.py

Co-authored-by: Chris Brozdowski &lt;CBrozdowski@yahoo.com&gt; ([`61c739f`](https://github.com/datajoint/element-array-ephys/commit/61c739f0ca0c6cfb74b204919485cf36dc417a8c))

* update docstrings ([`816a0ed`](https://github.com/datajoint/element-array-ephys/commit/816a0ed2322cda25e32b8171499b6de327ca9d98))

* figures downloaded to temp folder ([`643637d`](https://github.com/datajoint/element-array-ephys/commit/643637d816bb69f4faf792b0c8b31718ec670cfe))

* Merge pull request #117 from ttngu207/main

remove typing - keep consistency and maintain backward compatibility prior to python 3.9 ([`678cd95`](https://github.com/datajoint/element-array-ephys/commit/678cd951928d9e499f81252336dfcc47966e10f5))

* remove typing - keep consistency and maintain backward compatibility prior to python 3.9 ([`44a2582`](https://github.com/datajoint/element-array-ephys/commit/44a2582e0dbdb6b820101676de4cad41b6deb9ec))

* Merge pull request #114 from ttngu207/main

Various fixes and improvements - no new feature ([`3635a4a`](https://github.com/datajoint/element-array-ephys/commit/3635a4abc1cc8fd7d82b0ca6c09ecd9e66a4be5d))

* add missing CHANGELOG link ([`c2c3482`](https://github.com/datajoint/element-array-ephys/commit/c2c34828c321ca141e1331db5e7f5da408364ba4))

* BLACK formatting ([`cbc3b62`](https://github.com/datajoint/element-array-ephys/commit/cbc3b6286fcb9bd0b13bc803e7739c1b0b9dfd34))

* BLACK formatting ([`6b375f9`](https://github.com/datajoint/element-array-ephys/commit/6b375f998089114c051a829d9541d8acfa8a5fbe))

* Update CHANGELOG.md ([`72e784b`](https://github.com/datajoint/element-array-ephys/commit/72e784b6a2e202da18024f5026df002768901cbf))

* BLACK formatting ([`b6ce2f7`](https://github.com/datajoint/element-array-ephys/commit/b6ce2f7dc5d3fb37f3f277399794c095d58ebc0a))

* Merge branch &#39;datajoint:main&#39; into main ([`731c103`](https://github.com/datajoint/element-array-ephys/commit/731c10313c4b01da9bfb440227d48c4118600dd0))

* Merge pull request #115 from iamamutt/main

Remove ibllib dependency ([`561df39`](https://github.com/datajoint/element-array-ephys/commit/561df399a01a113346b8b2c9619fc7f98b953414))

* Merge pull request #1 from JaerongA/ephys_test

fix module name &amp; add docstrings ([`dd6e215`](https://github.com/datajoint/element-array-ephys/commit/dd6e215df2072aece593d2a8d02e67d3fed3fd47))

* Update CHANGELOG.md

Co-authored-by: Chris Brozdowski &lt;CBrozdowski@yahoo.com&gt; ([`ed5bbb9`](https://github.com/datajoint/element-array-ephys/commit/ed5bbb9e86d09870a5497a035de7e3eddf84d74c))

* update changelog ([`d07a93f`](https://github.com/datajoint/element-array-ephys/commit/d07a93ff5f8ca734847a412cbad4747e4c861383))

* bugfix - fix &#34;probe_indices&#34; in single probe recording ([`2676a16`](https://github.com/datajoint/element-array-ephys/commit/2676a161325f18dae13be73849556557a8cea79d))

* handles single probe recording in &#34;Neuropix-PXI&#34; format ([`1859085`](https://github.com/datajoint/element-array-ephys/commit/1859085e7133f5971862e0c74eb1176b83e7e426))

* safeguard in creating/inserting probe types upon `probe` activation ([`c2d9f47`](https://github.com/datajoint/element-array-ephys/commit/c2d9f47c0871b16ad82570fdd80ef802011f97b8))

* bugfix logging median subtraction duration ([`9ec904f`](https://github.com/datajoint/element-array-ephys/commit/9ec904f6697e4cddca068884613e42bbda092098))

* bugfix in updating median-subtraction duration ([`8ec0f71`](https://github.com/datajoint/element-array-ephys/commit/8ec0f713e461678c3288bfec083a0f762e885651))

* update duration for `median_subtraction` step ([`bd2ff1c`](https://github.com/datajoint/element-array-ephys/commit/bd2ff1cfe25bbc32c7b5ffc63add008d34bdd655))

* update docstring ([`68fa77c`](https://github.com/datajoint/element-array-ephys/commit/68fa77c2ed49c9603bccd64fcb75b92ac5b642e8))

* Apply suggestions from code review

Co-authored-by: Thinh Nguyen &lt;thinh@datajoint.com&gt; ([`cd9501c`](https://github.com/datajoint/element-array-ephys/commit/cd9501c1c773df08f4eea740ef312431e9ec5a1c))

* fix docstring &amp; formatting ([`9fc7477`](https://github.com/datajoint/element-array-ephys/commit/9fc7477abc7b975c3435d54a83063a248c50d42e))

* fix docstring in probe.py ([`7958727`](https://github.com/datajoint/element-array-ephys/commit/7958727e42bd5e235112fd3d14ef3435d8c6dcc5))

* feat: ([`ff3fca0`](https://github.com/datajoint/element-array-ephys/commit/ff3fca0ddc94f3921a06d8593005a2115ccdb930))

* remove proj() ([`496c210`](https://github.com/datajoint/element-array-ephys/commit/496c210a14acf6d5de4f3b67049a1af8738579cf))

* revert: :adhesive_bandage: revert to um ([`5a7f068`](https://github.com/datajoint/element-array-ephys/commit/5a7f06868a3ddff6616e7010d2cbceae944544aa))

* add probe_type in electrode_layouts ([`633f745`](https://github.com/datajoint/element-array-ephys/commit/633f7455c8dba47986b361380af6e73b88595b1b))

* spacing defaults to none ([`8f6e280`](https://github.com/datajoint/element-array-ephys/commit/8f6e28083d2e132eadef0e3cdb2b625fb43077bf))

* Update element_array_ephys/probe.py

Co-authored-by: Dimitri Yatsenko &lt;dimitri@datajoint.com&gt; ([`704f6ed`](https://github.com/datajoint/element-array-ephys/commit/704f6ed65c818037d33617cb424157f0d305fa5f))

* remove mapping dict ([`48ab889`](https://github.com/datajoint/element-array-ephys/commit/48ab889c0f3ac48813065d2243346fdf0e23e41d))

* col_count_per_shank ([`be3bd11`](https://github.com/datajoint/element-array-ephys/commit/be3bd11f720822565b59d80278f2eaa8b2cfc6cb))

* site_count_per_shank ([`cb21f61`](https://github.com/datajoint/element-array-ephys/commit/cb21f6154d737d2b59b2b5fec2690a8a247bad6f))

* modify build_electrodes function ([`1c7160c`](https://github.com/datajoint/element-array-ephys/commit/1c7160c33b1e316e49826a58dcf842790a18c94a))

* Merge branch &#39;main&#39; of https://github.com/iamamutt/element-array-ephys into ephys_test ([`8a65635`](https://github.com/datajoint/element-array-ephys/commit/8a65635dc78952e4227ae8c89a168dcc09a2b192))

* Merge remote-tracking branch &#39;upstream/main&#39; ([`b3a07b8`](https://github.com/datajoint/element-array-ephys/commit/b3a07b82d32f2d89d3a72ff4523e46c0518be20c))

* Merge pull request #113 from tdincer/main

Update README.md ([`85a1f0a`](https://github.com/datajoint/element-array-ephys/commit/85a1f0a238102122a0a38aed450648ce8477e4b6))

* Update README.md ([`7a5f843`](https://github.com/datajoint/element-array-ephys/commit/7a5f843568f0ebcc86681b8d802b494087d6e520))

* Merge branch &#39;datajoint:main&#39; into main ([`8dd5f29`](https://github.com/datajoint/element-array-ephys/commit/8dd5f29d4ce3070e430a31b8ab1f20cb800742b4))

* Merge pull request #111 from kabilar/main

Fix for cicd and other ux fixes ([`2e63edc`](https://github.com/datajoint/element-array-ephys/commit/2e63edc39de704ed77650f21643d50f54c433214))

* Move datajoint diagrams ([`8f006c0`](https://github.com/datajoint/element-array-ephys/commit/8f006c040fd82c4250e58bce220f064ff640aca2))

* Remove empty spaces to compare with ephys modules ([`863d9b1`](https://github.com/datajoint/element-array-ephys/commit/863d9b1152525dbb129c71dab40e2e22183a06d1))

* Fix bug ([`8731def`](https://github.com/datajoint/element-array-ephys/commit/8731def1d3edc1ba1f6d1735984c05d93903b35d))

* Fix diagram ([`566bc64`](https://github.com/datajoint/element-array-ephys/commit/566bc64d739fb0ec5f62b01887284aa6988f198b))

* Update text ([`48900e8`](https://github.com/datajoint/element-array-ephys/commit/48900e87cae10f1daf2d2f8e3e33520354b65f88))

* Merge pull request #110 from CBroz1/docs2

Add diagram text layer ([`34912bf`](https://github.com/datajoint/element-array-ephys/commit/34912bf062f738c9535042a8d9a55f5f1d3c74a5))

* Add diagram text layer ([`638ebc4`](https://github.com/datajoint/element-array-ephys/commit/638ebc4f23b9f1a7a5edc8c039a0e24dac30cc1d))

* Merge pull request #109 from CBroz1/docs2

Docs2 ([`23bf669`](https://github.com/datajoint/element-array-ephys/commit/23bf66956ca094e053ff3f8b22b612f0e018d6d7))

* Update diagram ([`07f0733`](https://github.com/datajoint/element-array-ephys/commit/07f0733ef00c0bf855c886a9571336768b8f51ff))

* Add diagram ([`38cc7ab`](https://github.com/datajoint/element-array-ephys/commit/38cc7ab4d0678d60cac03729332f6c548fc4c4fd))

* Update logo/styling. Hard wrap ([`4d22a16`](https://github.com/datajoint/element-array-ephys/commit/4d22a169094c437b7a80a92c51b92651ec3d5042))

* datatype clarification ([`c353400`](https://github.com/datajoint/element-array-ephys/commit/c353400e967e942e81148f60ac77c792ff68eccf))

* fix docstring typo ([`50d3dd1`](https://github.com/datajoint/element-array-ephys/commit/50d3dd1fcfe6a4ffd75933e04c5e7cb28564b83e))

* Merge pull request #107 from kushalbakshi/main

Updated CHANGELOG ([`2af7fc5`](https://github.com/datajoint/element-array-ephys/commit/2af7fc55a1f5348694a53c39cc338855f2bf5ae2))

## v0.2.0 (2022-11-03)

### Feature

* feat: :sparkles: Merge branch &#39;plotly&#39; into no_curation_plot ([`06c1064`](https://github.com/datajoint/element-array-ephys/commit/06c1064dd890d68afc90a9cc3aca3961edab2691))

* feat: :sparkles: add a report schema and plotting png figures ([`66743cc`](https://github.com/datajoint/element-array-ephys/commit/66743cc3dcdc22a35ddd24f7c694278c2903957a))

### Fix

* fix: :bug: use to_plotly_json() instead of to_json() ([`69b2796`](https://github.com/datajoint/element-array-ephys/commit/69b2796285690bcb68d2e3185608a6e33172c0ea))

### Unknown

* Updated CHANGELOG ([`bc5afcc`](https://github.com/datajoint/element-array-ephys/commit/bc5afcc97b23b8032722985158e69b6b01eb34f2))

* Merge pull request #102 from kushalbakshi/main

Added docs + docstrings ([`e04841b`](https://github.com/datajoint/element-array-ephys/commit/e04841b965700551046b29efd98d27577a5c4495))

* Update element_array_ephys/ephys_precluster.py

Co-authored-by: Chris Brozdowski &lt;CBrozdowski@yahoo.com&gt; ([`a73fcd2`](https://github.com/datajoint/element-array-ephys/commit/a73fcd2df9ee12860b2d548a45118ac6f5384b51))

* Update element_array_ephys/ephys_no_curation.py

Co-authored-by: Chris Brozdowski &lt;CBrozdowski@yahoo.com&gt; ([`36b8161`](https://github.com/datajoint/element-array-ephys/commit/36b8161359cded22c65ca863ebf22327652aeffd))

* Update element_array_ephys/ephys_chronic.py

Co-authored-by: Chris Brozdowski &lt;CBrozdowski@yahoo.com&gt; ([`e6e0b21`](https://github.com/datajoint/element-array-ephys/commit/e6e0b219924416df1012a3bb474f7c31f0207c61))

* Update element_array_ephys/ephys_acute.py

Co-authored-by: Chris Brozdowski &lt;CBrozdowski@yahoo.com&gt; ([`f2d550e`](https://github.com/datajoint/element-array-ephys/commit/f2d550e6903bf3e48802cb705bcccd48d1f9f765))

* Version and CHANGELOG 0.2.1 -&gt; 0.2.0 ([`2e0cffe`](https://github.com/datajoint/element-array-ephys/commit/2e0cffefc57a2ef2fd7fcb84c1a41796cd31d7bd))

* update CHANGELOG ([`473ca98`](https://github.com/datajoint/element-array-ephys/commit/473ca98f2ecfc351fd17ab5d21bf35d25a269bfb))

* Updated CHANGELOG and version ([`8b4f4fa`](https://github.com/datajoint/element-array-ephys/commit/8b4f4fac82b6ccb3d95332d62b1a8b317139f8cc))

* Merge branch &#39;main&#39; of https://github.com/datajoint/element-array-ephys ([`dc52a6e`](https://github.com/datajoint/element-array-ephys/commit/dc52a6eba8944aecc239e254d1b421079213bab0))

* Merge pull request #106 from ttngu207/main

add to changelog, bump version ([`89f1d7c`](https://github.com/datajoint/element-array-ephys/commit/89f1d7c7ffdcf49e52548222554a64de96f3e2ea))

* Apply suggestions from code review

Co-authored-by: Kabilar Gunalan &lt;kabilar@datajoint.com&gt;
Co-authored-by: Tolga Dincer &lt;tolgadincer@gmail.com&gt; ([`8f3e76e`](https://github.com/datajoint/element-array-ephys/commit/8f3e76edf254f18be9235d378ad5f839d50a4a51))

* add to changelog, bump version ([`02fb576`](https://github.com/datajoint/element-array-ephys/commit/02fb5765d6f71ba558d2e22fc72e970076a875c8))

* Merge pull request #103 from JaerongA/no_curation_plot

add ephys_report schema for data visualizations ([`3e9b675`](https://github.com/datajoint/element-array-ephys/commit/3e9b675581ed16b2704836a94cf1e55195e35f67))

* Update element_array_ephys/plotting/widget.py

Co-authored-by: Thinh Nguyen &lt;thinh@vathes.com&gt; ([`d0c6797`](https://github.com/datajoint/element-array-ephys/commit/d0c67970e18e435d3c7e298a0d59d6ce42536bee))

* f-string formatting ([`14a970f`](https://github.com/datajoint/element-array-ephys/commit/14a970f2e500288f54b33176905dd7037b85b962))

* improve clarity and reduce complexity for activating and using the `ephys_report` module ([`5d41039`](https://github.com/datajoint/element-array-ephys/commit/5d41039c7898ce73fa683de6d426636a6099121a))

* Merge branch &#39;no_curation_plot&#39; of https://github.com/JaerongA/element-array-ephys into no_curation_plot ([`d7fb6df`](https://github.com/datajoint/element-array-ephys/commit/d7fb6dff697ed7fa8f89b422f4621dca67fd5886))

* Apply suggestions from code review

Co-authored-by: Thinh Nguyen &lt;thinh@vathes.com&gt; ([`e773e94`](https://github.com/datajoint/element-array-ephys/commit/e773e94f22b9334b3586c8d7f7083afa699009bf))

* plot only the shank with the peak electrode ([`ce9abcc`](https://github.com/datajoint/element-array-ephys/commit/ce9abcc16ad02c0b54e122e4ed3c72cb33b1074c))

* new method for getting x, y spacing between sites ([`2eb9540`](https://github.com/datajoint/element-array-ephys/commit/2eb9540a436d08d35f1fd9cd7df205ef3a126afe))

* fix code to calculate site spacing &amp; figure reformatting ([`b48aaf0`](https://github.com/datajoint/element-array-ephys/commit/b48aaf078390954ad5be39b29fc84fa6e7db340d))

* add vscode in .gitignore ([`48e0744`](https://github.com/datajoint/element-array-ephys/commit/48e07444f0239b6479cb27dc40d59300b11e0691))

* fixed typo &amp; black formatting ([`f1d6a87`](https://github.com/datajoint/element-array-ephys/commit/f1d6a87df42d89ed7b33acd299cddcd35c4d1f71))

* clean up import &amp; remove wrong documentation ([`dc9b293`](https://github.com/datajoint/element-array-ephys/commit/dc9b2932ea9a0cd9e87c57940d0960a5f84d15bc))

* remove zip function ([`357cda9`](https://github.com/datajoint/element-array-ephys/commit/357cda99405b2605dad4d5b76142de8d0c9db6f2))

* remove zip function ([`290feca`](https://github.com/datajoint/element-array-ephys/commit/290fecab3299a32eb1650b7f58090027f6fab87e))

* add skip_duplicates=True in probe.Probe.insert ([`dab5dfe`](https://github.com/datajoint/element-array-ephys/commit/dab5dfe0e89f9cc668ce024403839c7b21ae0e0f))

* Apply suggestions from code review

Co-authored-by: Dimitri Yatsenko &lt;dimitri@datajoint.com&gt; ([`5d50894`](https://github.com/datajoint/element-array-ephys/commit/5d5089499c9fce8a260b3a51381fa3a254e48ec6))

* Update element_array_ephys/ephys_chronic.py

Co-authored-by: Dimitri Yatsenko &lt;dimitri@datajoint.com&gt; ([`35d2044`](https://github.com/datajoint/element-array-ephys/commit/35d2044abe97fbcf3351c01b6088007382a74f8d))

* pip plotly version ([`0e86529`](https://github.com/datajoint/element-array-ephys/commit/0e8652921ef7f97c64376dcc3367f3ec1e68cd7c))

* widget takes the activated ephys schema as an input ([`e302cf8`](https://github.com/datajoint/element-array-ephys/commit/e302cf87cf98520d467808998ae548ae44580676))

* add ephys widget ([`fb48db6`](https://github.com/datajoint/element-array-ephys/commit/fb48db69b07c15e05b3ff9a3ebd2c0e7890a8c58))

* update unit &amp; probe widget to be on the same widget ([`a41d7ba`](https://github.com/datajoint/element-array-ephys/commit/a41d7ba6c8567727ef5d3ba9edea48cddcfe525a))

* add widget event handler ([`d2b07d5`](https://github.com/datajoint/element-array-ephys/commit/d2b07d531ab2ac91f496b604084d4f5fe7279424))

* add ipywidget ([`cdaa931`](https://github.com/datajoint/element-array-ephys/commit/cdaa931804356bd71f7e4642e2e380a5efb19b69))

* adjust the figure size ([`00052e5`](https://github.com/datajoint/element-array-ephys/commit/00052e5746351a1194a861c4524d12eb0579e93b))

* fix naming convention ([`aab4ead`](https://github.com/datajoint/element-array-ephys/commit/aab4eadb914fa9302615ead69d8ef08545fa8775))

* update the probe widget ([`b3c3f5b`](https://github.com/datajoint/element-array-ephys/commit/b3c3f5bf9249105182c9bea10e392cb05ad1011b))

* update dependencies ([`75e15d2`](https://github.com/datajoint/element-array-ephys/commit/75e15d2f3154eeca5e69c95bb5a3961a1c657a5a))

* Merge branch &#39;main&#39; of https://github.com/datajoint/element-array-ephys into no_curation_plot ([`5027d17`](https://github.com/datajoint/element-array-ephys/commit/5027d17537c131d6e4959d5fad2c7e1466ca5db3))

* resolve circular dependency &amp; reformatting ([`2b69ab5`](https://github.com/datajoint/element-array-ephys/commit/2b69ab53e5415813414c967342f50ed1afb00b30))

* add shank in ProbeLevelReport &amp; formatting ([`36ce21f`](https://github.com/datajoint/element-array-ephys/commit/36ce21f3809b2a890d0603a8a054dad77ddf6e1c))

* Merge pull request #1 from ttngu207/ephys_report

resolve &#34;activation&#34; and circular dependency ([`f191214`](https://github.com/datajoint/element-array-ephys/commit/f19121470d02ebe9f0ca77f4e09f9e3bb71ddd8e))

* Update element_array_ephys/plotting/unit_level.py ([`36f128a`](https://github.com/datajoint/element-array-ephys/commit/36f128a411853e6e96614aac372b5c392ce307e0))

* resolve &#34;activation&#34; and circular dependency ([`4693728`](https://github.com/datajoint/element-array-ephys/commit/46937287c9c4b05efd2f6d121b6ea0142a2aa845))

* change report to ephys_report ([`f7ea1b8`](https://github.com/datajoint/element-array-ephys/commit/f7ea1b81281be6c39a39e1e8900c291b439de920))

* convert the unit report figures to plotly ([`275b479`](https://github.com/datajoint/element-array-ephys/commit/275b479da06a7ac7b5a188f6d1e57cf39a2bee69))

* Merge branch &#39;main&#39; into no-curation ([`6af6206`](https://github.com/datajoint/element-array-ephys/commit/6af6206e7a14adddbde98288a4e6c460a19516bb))

* Merge branch &#39;main&#39; of https://github.com/kushalbakshi/element-array-ephys ([`ab7b78c`](https://github.com/datajoint/element-array-ephys/commit/ab7b78c1dea17e173d0365c647066d1a02c2e22f))

* Update element_array_ephys/ephys_precluster.py

Co-authored-by: Chris Brozdowski &lt;CBrozdowski@yahoo.com&gt; ([`f6b93d7`](https://github.com/datajoint/element-array-ephys/commit/f6b93d75768053636cbc495a1600a19b9535f0b7))

* Update element_array_ephys/ephys_no_curation.py

Co-authored-by: Chris Brozdowski &lt;CBrozdowski@yahoo.com&gt; ([`486d938`](https://github.com/datajoint/element-array-ephys/commit/486d938a5b06913927d5eef1c593b0ee8365153f))

* Update element_array_ephys/ephys_chronic.py

Co-authored-by: Chris Brozdowski &lt;CBrozdowski@yahoo.com&gt; ([`ab1a067`](https://github.com/datajoint/element-array-ephys/commit/ab1a0675b52d29abd9900d4286041d49e35e953a))

* Roadmap updated in concepts.md ([`aec85cc`](https://github.com/datajoint/element-array-ephys/commit/aec85cc35f3a17b217a13012041fcb5c4492e75d))

* `enum` attribute description updated ([`4f28ad1`](https://github.com/datajoint/element-array-ephys/commit/4f28ad15a10449d19bcafb98e5b592521ac8769d))

* Merge branch &#39;main&#39; of https://github.com/kushalbakshi/element-array-ephys ([`c5d4882`](https://github.com/datajoint/element-array-ephys/commit/c5d488239edd0fc3490c41c86108009662aff629))

* Update docs/mkdocs.yaml

Co-authored-by: Chris Brozdowski &lt;CBrozdowski@yahoo.com&gt; ([`2355a46`](https://github.com/datajoint/element-array-ephys/commit/2355a465c6ee5611c0a8b04339af45d782b48496))

* Sentence case in concepts.md ([`72024ea`](https://github.com/datajoint/element-array-ephys/commit/72024ea3ea4de13738244736b710d08cdc00014b))

* References added to concepts + version change ([`2528e6d`](https://github.com/datajoint/element-array-ephys/commit/2528e6dea5788f5929b31eebc86930da6c03d44d))

* Minor formatting update to docstrings ([`5083b78`](https://github.com/datajoint/element-array-ephys/commit/5083b78b2ad78c852d5bea4747a95a16f5b663e8))

* Merge branch &#39;main&#39; of https://github.com/kushalbakshi/element-array-ephys ([`b030951`](https://github.com/datajoint/element-array-ephys/commit/b03095182e50e8700a7f78f815db9b7df401ad2e))

* Update element_array_ephys/ephys_precluster.py

Co-authored-by: Chris Brozdowski &lt;CBrozdowski@yahoo.com&gt; ([`38f2410`](https://github.com/datajoint/element-array-ephys/commit/38f2410bed0f98de478d7f1795a08e065bedd4eb))

* Update element_array_ephys/ephys_precluster.py

Co-authored-by: Chris Brozdowski &lt;CBrozdowski@yahoo.com&gt; ([`ca1441f`](https://github.com/datajoint/element-array-ephys/commit/ca1441f24446c89e63d708deef9318de6f83a543))

* Update element_array_ephys/probe.py

Co-authored-by: Chris Brozdowski &lt;CBrozdowski@yahoo.com&gt; ([`98f8a8c`](https://github.com/datajoint/element-array-ephys/commit/98f8a8c2e62a06baaa80e61d634cff7bf25acc0b))

* Updated docstrings after code review ([`17d9e4a`](https://github.com/datajoint/element-array-ephys/commit/17d9e4aed104ee047c77557591c187259ef6575a))

* Changes applied from code review ([`c75e2d7`](https://github.com/datajoint/element-array-ephys/commit/c75e2d71e485336050ad94a1016bd9af1cc0e432))

* Update docs/mkdocs.yaml

Co-authored-by: Chris Brozdowski &lt;CBrozdowski@yahoo.com&gt; ([`37d6a60`](https://github.com/datajoint/element-array-ephys/commit/37d6a6098632044474a9c2caae97ece16c4b2c68))

* Automated versioning added ([`7431058`](https://github.com/datajoint/element-array-ephys/commit/74310589f076f2f44283c0d3ff3cd46ac0de54e0))

* science-team to concepts + hard wrap test ([`b0e755f`](https://github.com/datajoint/element-array-ephys/commit/b0e755f6cc14313eb83e2a2e09524f751d0c5ff0))

* Docstrings added ([`0feee75`](https://github.com/datajoint/element-array-ephys/commit/0feee75270d8a4a6e99507937f2aa530bd4b7dcd))

* Updated docs based on DLC merge ([`01322db`](https://github.com/datajoint/element-array-ephys/commit/01322db39dc06a540799142788c3173287323347))

* Updates mirroring DLC ([`0111dcd`](https://github.com/datajoint/element-array-ephys/commit/0111dcdf3ab3e7e1d3b95a2e4a8fcc7ac3ca3cef))

* Fixes to docs after local testing + docstrings ([`7135ce1`](https://github.com/datajoint/element-array-ephys/commit/7135ce19de7080b5c4fe11f1785f3ce46433c924))

* Updated docstrings + Dockerfiles ([`e7423e2`](https://github.com/datajoint/element-array-ephys/commit/e7423e2d330c17a2e68f7aee4f06237495d30fd3))

* Merge branch &#39;main&#39; of https://github.com/datajoint/element-array-ephys ([`afb64eb`](https://github.com/datajoint/element-array-ephys/commit/afb64ebc93d33a3aa06407e7b96f98925f4c5fad))

* Merge pull request #94 from datajoint/run_kilosort

`run_kilosort` -&gt; `main` ([`db75e4d`](https://github.com/datajoint/element-array-ephys/commit/db75e4dbd3770ce54b43d16ed652e338d440eab2))

* Merge pull request #97 from ttngu207/no-curation

pull from main - add QC ([`03e2d5f`](https://github.com/datajoint/element-array-ephys/commit/03e2d5fcfbfa2daa15d2962676903ede80daa497))

* specify lfp filepath as input ([`10d12a6`](https://github.com/datajoint/element-array-ephys/commit/10d12a696057ef44da373cbad0029350fc60761e))

* smart handling of finished median subtraction step ([`34e59cc`](https://github.com/datajoint/element-array-ephys/commit/34e59cc20ca31472ddba07e190c940ca432b6b99))

* modify `extracted_data_directory` path - same as ks output path ([`296f7c6`](https://github.com/datajoint/element-array-ephys/commit/296f7c672296a2502eb84bbc03e7138e42da00bf))

* bugfix QC ingestion ([`2d76efc`](https://github.com/datajoint/element-array-ephys/commit/2d76efcccb4f47569a4de9f4c4a2ca215b45146c))

* bugfix - remove `%` in attributes&#39; comments ([`d008b05`](https://github.com/datajoint/element-array-ephys/commit/d008b051f0d4752b9582d642e5f948d84386b902))

* add QC to `ephys_no_curation` ([`db448f7`](https://github.com/datajoint/element-array-ephys/commit/db448f72a1f7bc091edfd3c29dabadeea71c6d24))

* Merge branch &#39;main&#39; of https://github.com/datajoint/element-array-ephys into no-curation ([`ef486ed`](https://github.com/datajoint/element-array-ephys/commit/ef486ed2e686ed1c59f4364d8570dc52c96c2347))

* median subtraction on a copied data file ([`32bff24`](https://github.com/datajoint/element-array-ephys/commit/32bff24a9f325673fc8606fee2d916a894000e80))

* Merge pull request #93 from CBroz1/rk

Ensure Path type for get_spikeglx_meta_filepath ([`a738ee7`](https://github.com/datajoint/element-array-ephys/commit/a738ee74a4bd593a43332676ed5bcfa005434319))

* Ensure Path type for get_spikeglx_meta_filepath ([`0e94252`](https://github.com/datajoint/element-array-ephys/commit/0e942523d300ac01d56ae8419cb62e696d3abae2))

* Merge pull request #91 from ttngu207/no-curation

bugfix for catgt ([`6757ef7`](https://github.com/datajoint/element-array-ephys/commit/6757ef738b95ad8c7cc7e305a7a9a0f96de42fe9))

* Update kilosort_triggering.py ([`71d87ae`](https://github.com/datajoint/element-array-ephys/commit/71d87aecfe37d6a435f5cc819ee389ed621d9772))

* Merge pull request #90 from ttngu207/no-curation

enable `CatGT` ([`23ca7ca`](https://github.com/datajoint/element-array-ephys/commit/23ca7ca6d2474daab71d8a00c4c492d4395a2667))

* improve error handling ([`f60ba3d`](https://github.com/datajoint/element-array-ephys/commit/f60ba3d6a0c0776d1f7b97d02b9d1608a5788a0e))

* bugfix - path search for catgt output ([`c33d1b0`](https://github.com/datajoint/element-array-ephys/commit/c33d1b02811dbaf7e492da60e70819d5102798ff))

* `missing_ok` arg only available in python 3.8+ ([`35da39b`](https://github.com/datajoint/element-array-ephys/commit/35da39bb1c5c80c18ade40fed296cfb47b158123))

* bugfix ([`fbdbe24`](https://github.com/datajoint/element-array-ephys/commit/fbdbe24adc55c6fb51f4e524760528f27589f37b))

* bugfix ([`aed42ca`](https://github.com/datajoint/element-array-ephys/commit/aed42ca1dddd15a6b2a03c329f9278122846c55b))

* enable catgt run ([`987231b`](https://github.com/datajoint/element-array-ephys/commit/987231be7d386351a043901d19f55f76b6bbf90d))

* bugfix in running catgt ([`5905392`](https://github.com/datajoint/element-array-ephys/commit/59053923a9968df6f2ab7f90d9324c7502dcaf0e))

* Update kilosort_triggering.py ([`f9f18d0`](https://github.com/datajoint/element-array-ephys/commit/f9f18d0864615f24420b3feb982b68dfd85dd6d7))

* Merge pull request #89 from ttngu207/no-curation

implement data compression using `mtscomp` for openephys and spikeglx for neuropixels data ([`c153e7f`](https://github.com/datajoint/element-array-ephys/commit/c153e7f4f0875a08f097e6af653bb6e8248c5c08))

* garbage collect openephys objects ([`d8aea04`](https://github.com/datajoint/element-array-ephys/commit/d8aea041f35ba2b92e8f619f4b3729123852116a))

* garbage collect openephys objects ([`97f3d21`](https://github.com/datajoint/element-array-ephys/commit/97f3d21ce0ae6e5827070d093893ab836665998a))

* implement data compression using `mtscomp` for openephys and spikeglx neuropixels data ([`b2bd0ee`](https://github.com/datajoint/element-array-ephys/commit/b2bd0eeab31a63d95fcaf84aaafb436289da8838))

* Merge pull request #88 from ttngu207/no-curation

overall code cleanup/improvement for more robust and optimal kilosort run ([`ad8436e`](https://github.com/datajoint/element-array-ephys/commit/ad8436e8535ab34fdb24efea7e0aa9bc5d2d6178))

* Merge branch &#39;no-curation&#39; of https://github.com/ttngu207/element-array-ephys into no-curation ([`fd331bd`](https://github.com/datajoint/element-array-ephys/commit/fd331bdefc036eb9e08fad83b8ffba41dc037ec7))

* remove space escaping character ([`b71b459`](https://github.com/datajoint/element-array-ephys/commit/b71b459744b212251d0685b7bebb82d859fc8723))

* improve kilosort calls, handle spaces in paths ([`0c77826`](https://github.com/datajoint/element-array-ephys/commit/0c77826af1141d0e2d5828736252b33e56734af5))

* improve error message ([`a3c5c2f`](https://github.com/datajoint/element-array-ephys/commit/a3c5c2fb9c03e3b6df293ed0e8fb58f17a20ef78))

* bugfix, match new implementation for openephys ([`3f1ee37`](https://github.com/datajoint/element-array-ephys/commit/3f1ee371bec5c68a2c9838082df87b6368074ebd))

* code cleanup, minor bugfix ([`b97566e`](https://github.com/datajoint/element-array-ephys/commit/b97566e6b833d610377e93cd21a08a0272f3a075))

* improve logic for running kilosort modules in a resumable fashion ([`9a59e57`](https://github.com/datajoint/element-array-ephys/commit/9a59e574dd25176a8da9b8142d6e87aeed3c5f74))

* Merge pull request #86 from CBroz1/rk

Changes for codebook deployment ([`d9c3887`](https://github.com/datajoint/element-array-ephys/commit/d9c38873e3371dd045bb90e199a0a58caa5e701b))

* WIP: version bump pynwb to 2.0 ([`0221848`](https://github.com/datajoint/element-array-ephys/commit/0221848b6466daba6553e1e1a5967a2dee08c954))

* Merge branch &#39;run_kilosort&#39; of https://github.com/datajoint/element-array-ephys into rk ([`13d74ad`](https://github.com/datajoint/element-array-ephys/commit/13d74ad73efac3cd5f94f1ffe374c762a95c936d))

* Merge pull request #77 from ttngu207/no-curation

more robust loading of openephys format ([`d298b07`](https://github.com/datajoint/element-array-ephys/commit/d298b07c0ba7805a5efeee2d9db703cc35913925))

* more robust loading of openephys format ([`67039ac`](https://github.com/datajoint/element-array-ephys/commit/67039ac51bd87754a610053bf8e85d0a958f42be))

* Merge pull request #73 from ttngu207/no-curation

update openephys loader - handling open ephys v0.6.0 ([`9272ee6`](https://github.com/datajoint/element-array-ephys/commit/9272ee64417248b7afe85786ae60366710c0f0ff))

* added loading of electrode location for new openephys format ([`4e367d7`](https://github.com/datajoint/element-array-ephys/commit/4e367d72dc9928b69553613844773ad38c98ea91))

* update open ephys loader to handle &#34;STREAM&#34; in latest format ([`07604e2`](https://github.com/datajoint/element-array-ephys/commit/07604e24c185421566122024b3d8b8c3f60b4475))

* Merge pull request #70 from ttngu207/no-curation

bugfix for LFP electrode mapping ([`747c15f`](https://github.com/datajoint/element-array-ephys/commit/747c15f5b481790e6e36131dfee8ab932eeb6220))

* bugfix for LFP electrode mapping ([`f11e016`](https://github.com/datajoint/element-array-ephys/commit/f11e0161e949bc572f0ac7ee2e0b46096fa00351))

* `kilosort2` also as part of the `contents` for ClusteringMethod ([`f4b917d`](https://github.com/datajoint/element-array-ephys/commit/f4b917d0eada17b3c54c4922ef2295f368569855))

* Update requirements.txt ([`a578d85`](https://github.com/datajoint/element-array-ephys/commit/a578d851db1cc439b6c1bc380bdb6ee6d6af4789))

* Add contact info to Code of Conduct ([`70e0b1c`](https://github.com/datajoint/element-array-ephys/commit/70e0b1c3f18d7e150957e5ad943a9f3c76d130e3))

* Add Code of Conduct ([`e43e5d5`](https://github.com/datajoint/element-array-ephys/commit/e43e5d5b88ca82c1be14b7392c56397a9162b7c6))

* Issue #63 ([`d102f6f`](https://github.com/datajoint/element-array-ephys/commit/d102f6fa5bb79c8ddf23e3c6df4e200cb3c02a25))

* Merge branch &#39;rk&#39; of https://github.com/CBroz1/element-array-ephys into rk ([`bd6d7e4`](https://github.com/datajoint/element-array-ephys/commit/bd6d7e471ea322e4675dbf23bd253e53ac057ca1))

* Update README.md

Co-authored-by: Dimitri Yatsenko &lt;dimitri@datajoint.com&gt; ([`720e355`](https://github.com/datajoint/element-array-ephys/commit/720e355c9201f79d1638a15099e325ef0dda76cd))

* Issue #11 ([`4a3e0bf`](https://github.com/datajoint/element-array-ephys/commit/4a3e0bf0332abcab167e7fc65c99de3f98f39e2f))

* WIP: nwb line length, Readme mention of ([`e1c9b35`](https://github.com/datajoint/element-array-ephys/commit/e1c9b355d1270acd6ccd144106b173e68c0e6654))

* WIP: NWB fix - prevent SQL err by restricting key ([`b62fd12`](https://github.com/datajoint/element-array-ephys/commit/b62fd12f5d94ccd70aa5d816ac5c2320b7d8520d))

* WIP: nwb bugfix ([`49bba8a`](https://github.com/datajoint/element-array-ephys/commit/49bba8a2f03fd1e0a74910bf360dc1b68308f261))

* Merge pull request #69 from ttngu207/no-curation

Add no-curation version and run kilosort analysis ([`364f80e`](https://github.com/datajoint/element-array-ephys/commit/364f80ed261e6297c20b14a3915ba29b6beb7cb4))

* Merge remote-tracking branch &#39;upstream/run_kilosort&#39; into no-curation ([`ddd4095`](https://github.com/datajoint/element-array-ephys/commit/ddd409543cc531ef11c1961fb50ffc1a5a516772))

* Merge branch &#39;no-curation&#39; of https://github.com/ttngu207/element-array-ephys into no-curation ([`7fecff1`](https://github.com/datajoint/element-array-ephys/commit/7fecff10fd4b530af00e1e408b723fab93787fd4))

* Apply suggestions from code review

Co-authored-by: Dimitri Yatsenko &lt;dimitri@datajoint.com&gt; ([`af864d7`](https://github.com/datajoint/element-array-ephys/commit/af864d73d5925458d0955d6c0556366202ad88ee))

* added assertion - safeguard against failed loading of continuous.dat ([`47babf3`](https://github.com/datajoint/element-array-ephys/commit/47babf37c9556d1d496cd4d7687baea7d57cd7eb))

* handles new probe naming in latest Open Ephys format ([`cd5fe70`](https://github.com/datajoint/element-array-ephys/commit/cd5fe70261b48142f4406087eacdde96c141de99))

* update openephys loader - handle new open ephys format ([`11a12ba`](https://github.com/datajoint/element-array-ephys/commit/11a12ba00a169618d510f967132ab738f24151e4))

* Update openephys.py ([`85c7c8b`](https://github.com/datajoint/element-array-ephys/commit/85c7c8ba12154fff8695b1a66d9db96440e8bb08))

* configurable `paramset_idx` for auto ClusteringTask generation ([`39c8579`](https://github.com/datajoint/element-array-ephys/commit/39c8579821aec505fffaf40c8e41d833ec9f775f))

* bugfix ([`769de13`](https://github.com/datajoint/element-array-ephys/commit/769de136cfd91e3ae57228d657af405a302ebeaf))

* Update __init__.py ([`e0a9a4f`](https://github.com/datajoint/element-array-ephys/commit/e0a9a4f38fe74e790e1d1c70cf3b7ab1d68f8f8a))

* delete nwb export - rename `ephys` module -&gt; `ephys_acute` ([`c2f8aea`](https://github.com/datajoint/element-array-ephys/commit/c2f8aeaa7aad48c602484b9358a569d2b54d69c4))

* remove unintended prototyping work ([`83649f5`](https://github.com/datajoint/element-array-ephys/commit/83649f5edf2ddb4788e5b361ec7153238ecbfee4))

* handle older open ephys format for single probe ([`bdcfa46`](https://github.com/datajoint/element-array-ephys/commit/bdcfa46ea7d1d55af48b22556b9943c9d44b5fff))

* Update requirements.txt ([`f0b3d4a`](https://github.com/datajoint/element-array-ephys/commit/f0b3d4a7b6c77c6bb295ebb558458caacaa1543a))

* Update requirements.txt ([`7320f9f`](https://github.com/datajoint/element-array-ephys/commit/7320f9f6b966548145b4a39b5bbf87b0d5c8d6e3))

* Update requirements.txt ([`cb1a041`](https://github.com/datajoint/element-array-ephys/commit/cb1a0419d8707808b1f1e0599358be6c42a00bd2))

* rename `sess_dir` -&gt; `session_dir` ([`03cab02`](https://github.com/datajoint/element-array-ephys/commit/03cab02ee709e94b151621d00b12e455953dccfb))

* Apply suggestions from code review

Co-authored-by: Dimitri Yatsenko &lt;dimitri@datajoint.com&gt; ([`f4052cc`](https://github.com/datajoint/element-array-ephys/commit/f4052cc5d079e6ea065ce6f6aab34b462250ba39))

* name tweak ([`779b2fb`](https://github.com/datajoint/element-array-ephys/commit/779b2fb95b277768cb15046a4eea4e7138e23749))

* minor bugfix ([`d66368c`](https://github.com/datajoint/element-array-ephys/commit/d66368c7f016e244b4a81c1aa2f6b7d7bbc4d15d))

* Merge branch &#39;main&#39; of https://github.com/datajoint/element-array-ephys into no-curation ([`b4832ea`](https://github.com/datajoint/element-array-ephys/commit/b4832ea82416a7d76675811b2936c062185514f8))

* DEPRECATING NWB EXPORT ([`4951b39`](https://github.com/datajoint/element-array-ephys/commit/4951b396a1b97a9dda3aacdc974c23f64d8bfd9f))

* Merge pull request #4 from A-Baji/no-curation

all three ephys files up to date ([`9dd6b42`](https://github.com/datajoint/element-array-ephys/commit/9dd6b42fc77e18089a419e10457013c86380db0c))

* applied requested changes ([`c56cd18`](https://github.com/datajoint/element-array-ephys/commit/c56cd188357f84da89c72c56d74b6bdf6913f11e))

* all three ephys files up to date ([`f2881ca`](https://github.com/datajoint/element-array-ephys/commit/f2881ca0d2c61dfcdeff9794f6fe26b1ecb6a066))

* Merge pull request #3 from ttngu207/nwb-export

bugfix in assigning unit electrode indices ([`f9a4754`](https://github.com/datajoint/element-array-ephys/commit/f9a4754355fe556c6a7ae28f068556568a9966a1))

* bugfix in assigning unit electrode indices ([`31bba8c`](https://github.com/datajoint/element-array-ephys/commit/31bba8ca4061f6225ea111649929a0e2eb942615))

* include probe as part of the electrode_group name for uniqueness ([`aa47c8a`](https://github.com/datajoint/element-array-ephys/commit/aa47c8a761e1149fd386a1e2ca910670f24e18b7))

* version bump ([`8657d58`](https://github.com/datajoint/element-array-ephys/commit/8657d58557cc755292d1be4638b4f8bf3c4af1ed))

* fix NWB export - null `waveforms` - back to version 0.1.0b1 ([`dae36d1`](https://github.com/datajoint/element-array-ephys/commit/dae36d1f4793047b1a1ac684fc0f37f529d1c9aa))

* version bump ([`6107e8e`](https://github.com/datajoint/element-array-ephys/commit/6107e8e22f29b858a2ef139ea5890b7cfa86b80b))

* NWB export fix, specifying which ephys module ([`8dff08b`](https://github.com/datajoint/element-array-ephys/commit/8dff08b5c094c0867cea242738f8a9e1ffbca6ba))

* handles multi-probe for older OE version ([`02c4b67`](https://github.com/datajoint/element-array-ephys/commit/02c4b671c2f58aaf19a966af266d2e56d25f8a86))

* openephys loader - handles signalchain and processor as single element or list ([`2022e91`](https://github.com/datajoint/element-array-ephys/commit/2022e91079ecb021b5a3b0cc0771631206692c9c))

* for pykilosort&#39;s probe, provide both Nchan and NchanTOT ([`1c39568`](https://github.com/datajoint/element-array-ephys/commit/1c39568045b65ac4de3597e3d539311efdb033c1))

* handle missing `sample_rate` from pykilosort params.py ([`142459d`](https://github.com/datajoint/element-array-ephys/commit/142459d6f21d096b1490e301cfbbeb22ac370e6c))

* bugfix in triggering pykilosort ([`02069c9`](https://github.com/datajoint/element-array-ephys/commit/02069c94b8b088a2f16b58f5c7224f17a3920cd5))

* clusters extraction - check `cluster_group.tsv` and `cluster_KSLabel.tsv` ([`da10c66`](https://github.com/datajoint/element-array-ephys/commit/da10c66caf6a99f0a0f63f89c94136f5470983c7))

* handles extraction of `connected` channels for NP_PROBE format in OpenEphys ([`43d6614`](https://github.com/datajoint/element-array-ephys/commit/43d6614f8b4a36aebcb81ed49500d890f35de1f7))

* bugfix, timedelta as seconds ([`adffe34`](https://github.com/datajoint/element-array-ephys/commit/adffe34ea52ed2f0550a2063b843b72d67d90ef2))

* bugfix - extract recording datetime (instead of using experiment datetime) ([`c213325`](https://github.com/datajoint/element-array-ephys/commit/c21332543e23f647e0198e9eba0881f256e85a87))

* bugfix openephys loader ([`0d16e7e`](https://github.com/datajoint/element-array-ephys/commit/0d16e7ed61ec7911d97db7077b6f11498709cf73))

* search recording channels for Open Ephys based on channel names ([`d105419`](https://github.com/datajoint/element-array-ephys/commit/d1054195a63608f041a62dd49d9134fae80dc89b))

* bugfix in electrode sites design for Neuropixels UHD probe ([`f55a6a7`](https://github.com/datajoint/element-array-ephys/commit/f55a6a7dfd373efe1405aeea050a0f8fe0b9e6f8))

* supporting `neuropixels UHD` in `ephys.EphysRecording` ([`db3027b`](https://github.com/datajoint/element-array-ephys/commit/db3027b7b48eca7e281cfa93c23b546545457ed4))

* handles format differences between npx1 vs 3A ([`e325a30`](https://github.com/datajoint/element-array-ephys/commit/e325a30d1ab4879077d59d773b50cb05998168bd))

* fix package requirement formatting error ([`af2b18b`](https://github.com/datajoint/element-array-ephys/commit/af2b18ba88287f234ab0ade5c76210a57eed718b))

* update openephys loader ([`4250220`](https://github.com/datajoint/element-array-ephys/commit/4250220c7933f47135208c31a5d0e6c46a2d8518))

* minor bugfix in running pykilosort ([`b6f8f99`](https://github.com/datajoint/element-array-ephys/commit/b6f8f99dbc8b2bc56ca2f6484d1f8f09f8056944))

* using fork of pyopenephys ([`96931a4`](https://github.com/datajoint/element-array-ephys/commit/96931a4fdfcaebcee2853dac3835e7fdf954524f))

* use_C_waves=False for OpenEphys ([`81d99c8`](https://github.com/datajoint/element-array-ephys/commit/81d99c8c4c6901daa23d01e19e760d3b2d737a6f))

* first prototype for pykilosort ([`819ff19`](https://github.com/datajoint/element-array-ephys/commit/819ff193f326e950e6f06dc4ed4785a4ba96be0b))

* triggering kilosort analysis for open-ephys ([`df599fb`](https://github.com/datajoint/element-array-ephys/commit/df599fbe88e0ffc4694c33479c1247c22f66760e))

* add `neuropixels UHD` probe type ([`ddc3b94`](https://github.com/datajoint/element-array-ephys/commit/ddc3b9429b53ea6d7e5889171884522e7a05dbad))

* specify additional recording-info as part of the `params` ([`58b5984`](https://github.com/datajoint/element-array-ephys/commit/58b598473ad6c83c8966176d484a0f23c8056a6b))

* bugfix for running kilosort for Open Ephys data ([`199a2ba`](https://github.com/datajoint/element-array-ephys/commit/199a2baf43eadc1028961991173d5b010d31bc39))

* first prototype for running the ecephys_pipeline with OpenEphys ([`49ca0be`](https://github.com/datajoint/element-array-ephys/commit/49ca0beded17dd3d613b498223cbced3ce5480f1))

* add nwb export to `no-curation` ephys ([`b25f065`](https://github.com/datajoint/element-array-ephys/commit/b25f065f64735727ddae3e5e6ef907ba7368bfb9))

* Merge pull request #2 from ttngu207/nwb-export

Nwb export ([`3ebdf23`](https://github.com/datajoint/element-array-ephys/commit/3ebdf236cb6f20cf0458f21b4358ca5b8b13c958))

* Update nwb.py ([`19616ef`](https://github.com/datajoint/element-array-ephys/commit/19616ef695d2546bfd441b32ea0df4a668488392))

* handle NWB export with multiple curated clusterings from one session ([`d07f830`](https://github.com/datajoint/element-array-ephys/commit/d07f830dc9384193164919399fb57605b3ea96c7))

* added NWB export ([`f740aef`](https://github.com/datajoint/element-array-ephys/commit/f740aef79c0b87a0e3b951c58e36daf701705195))

* minor bugfix ([`09c1e60`](https://github.com/datajoint/element-array-ephys/commit/09c1e6072dc681898e5edf9e8e866e9519ac76bd))

* stylistic improvements, addressing code review comments ([`e8ffe17`](https://github.com/datajoint/element-array-ephys/commit/e8ffe17711ad66bbf5011aef9bcff3f7ed2afe76))

* Apply suggestions from code review

Co-authored-by: Dimitri Yatsenko &lt;dimitri@datajoint.com&gt; ([`74e3ead`](https://github.com/datajoint/element-array-ephys/commit/74e3eadc0c722bef43901f075434142314604077))

* check `ap.bin` file validity before triggering kilosort (based on filesize) ([`beaf765`](https://github.com/datajoint/element-array-ephys/commit/beaf7651cffa67e1bc8a10b19cad49dde8e6530e))

* duration for each module run ([`19b704b`](https://github.com/datajoint/element-array-ephys/commit/19b704b70af25ff8e3b40d28ac1858748007c9bb))

* bugfix logging for kilosort triggering ([`f34e95d`](https://github.com/datajoint/element-array-ephys/commit/f34e95dcda3ba8379cfe9e860277d69f1336db37))

* minor bugfix ([`55bec01`](https://github.com/datajoint/element-array-ephys/commit/55bec0122f077bbb4b8ac90516da6d0a78dc8630))

* stage tracking and resumable kilosort run ([`408532c`](https://github.com/datajoint/element-array-ephys/commit/408532cf9b685b9a57f59831ba3fd16d0982ea97))

* minor cleanup ([`dc7ddd9`](https://github.com/datajoint/element-array-ephys/commit/dc7ddd912de849b8b63335bf33e700a630d117db))

* improve clusteringtask and waveform ingestion routine ([`c2ee64f`](https://github.com/datajoint/element-array-ephys/commit/c2ee64f52e6ce1be7062584dbd90129bae6cb891))

* new version 0.1.0b1 ([`67341d8`](https://github.com/datajoint/element-array-ephys/commit/67341d8f5470622e6c4e58a0b7f3ae3989c281a7))

* Update kilosort.py ([`0f0c212`](https://github.com/datajoint/element-array-ephys/commit/0f0c21249de2c81cef480df219f3c005a62f3b78))

* bugfix - no dir created ([`044c389`](https://github.com/datajoint/element-array-ephys/commit/044c389fae07540621a36af6568a35199cb3006a))

* add debugging lines ([`b9f4e92`](https://github.com/datajoint/element-array-ephys/commit/b9f4e9208b80f113b70f66d46fd6d4424dde4ec0))

* log the folder creation ([`ae966aa`](https://github.com/datajoint/element-array-ephys/commit/ae966aa9f55d7b835995c9d5a2b1e39f03e3f4ea))

* bugfix, convert path to string ([`94aade7`](https://github.com/datajoint/element-array-ephys/commit/94aade7212775ccc275979e05d17cb195bb665de))

* bugfix ([`28c4452`](https://github.com/datajoint/element-array-ephys/commit/28c445279f310822260c5df78f906c7bf77a3764))

* updating `kilosort_repository` depending on which KSVer to be used ([`38c5be6`](https://github.com/datajoint/element-array-ephys/commit/38c5be6fd8d1d1225c61c28eecabaf311e605694))

* include `clustering_method` into the calculation of `param_set_hash` ([`acdab12`](https://github.com/datajoint/element-array-ephys/commit/acdab125acfc62792fa4fa18ad0ba403d16a5da8))

* make variable naming consistent ([`a0ea9f7`](https://github.com/datajoint/element-array-ephys/commit/a0ea9f70c8dec59c9d415bd9de2e219ea69d0e81))

* add kilosort 2.5 as default content ([`a6cae12`](https://github.com/datajoint/element-array-ephys/commit/a6cae1291534e929c451d422da7083241f3418f9))

* minor bugfix ([`69c5e51`](https://github.com/datajoint/element-array-ephys/commit/69c5e5144c4210bd2248c354b56c6ba1bc4f6a47))

* change default `noise_template_use_rf` to False ([`c593baf`](https://github.com/datajoint/element-array-ephys/commit/c593bafbac84334c1e388fe96817494085878aed))

* missing generate module json ([`375e437`](https://github.com/datajoint/element-array-ephys/commit/375e437861d33791147d2913a8ea94d8031c12d6))

* bugfix ([`d63561f`](https://github.com/datajoint/element-array-ephys/commit/d63561f74ff7dbf1bc87922f74b5f55ad0bd5cd6))

* handle cases where `fileTimeSecs` is not available ([`6788180`](https://github.com/datajoint/element-array-ephys/commit/6788180682f8d2ff4ee3bdc0a6a01dd61814c67f))

* bugfix in triggering ecephys_spike_sorting ([`6bf0eb1`](https://github.com/datajoint/element-array-ephys/commit/6bf0eb100e0e5480e8824644ff1c3b638e889c24))

* minor tweak/improvements in kilosort triggering ([`f699ce7`](https://github.com/datajoint/element-array-ephys/commit/f699ce7e3af7c387579f133c763046f6e55517f4))

* Update kilosort_trigger.py ([`dd01fd2`](https://github.com/datajoint/element-array-ephys/commit/dd01fd2158d3f09f314e705b2c08f5e1b4205085))

* flag to create spike sorting output dir ([`6c646bb`](https://github.com/datajoint/element-array-ephys/commit/6c646bbcc2437f00c8de4f99aa1d6738c5acd09f))

* fix missing `clustering_method` ([`5cdc994`](https://github.com/datajoint/element-array-ephys/commit/5cdc994f47387f7935c028a1fd38e59e7d4c31e3))

* handles a weird windows/unix path incompatibility (even with pathlib) ([`ba28637`](https://github.com/datajoint/element-array-ephys/commit/ba28637496fbea77207a81fae3e6a287c56b494a))

* Merge branch &#39;no-curation&#39; of https://github.com/ttngu207/element-array-ephys into no-curation ([`a24bd1a`](https://github.com/datajoint/element-array-ephys/commit/a24bd1a700d7d8fed61c6c0f5e51c2482cbc5bbf))

* Apply suggestions from code review

Co-authored-by: Dimitri Yatsenko &lt;dimitri@datajoint.com&gt; ([`149ef3f`](https://github.com/datajoint/element-array-ephys/commit/149ef3f4ab5294399d0553acec05b00b8d79226b))

* typo fix ([`9f98195`](https://github.com/datajoint/element-array-ephys/commit/9f981951d9222af132a51a85267a2c278f893f27))

* minor stylistic improvements ([`a9326ea`](https://github.com/datajoint/element-array-ephys/commit/a9326eaa015c46875829dbf92d6494aa5c0a3d85))

* remove `_required_packages_paths` ([`60105da`](https://github.com/datajoint/element-array-ephys/commit/60105da78fcc5e55be961c7c79fdb206b072de26))

* triggering Kilosort with ecephys_spike_sorting package ([`047bfa1`](https://github.com/datajoint/element-array-ephys/commit/047bfa1cd33fadefe42e9f395f119c97e894d1d1))

* automate ClusteringTask insertion ([`0d56456`](https://github.com/datajoint/element-array-ephys/commit/0d5645668f18d8b9011e46817bd0f31dda73a088))

* bugfix ([`a7e7554`](https://github.com/datajoint/element-array-ephys/commit/a7e755481215b67917b9678c5c795634db775c03))

* Update ephys_no_curation.py ([`70e93b3`](https://github.com/datajoint/element-array-ephys/commit/70e93b381c1f5fcb45b01a647b5be0fc4fbbbdf0))

* using `find_full_path` for session dir as well - improve robustness ([`5420ae0`](https://github.com/datajoint/element-array-ephys/commit/5420ae05ddf63c56e729f4229547fe5df99b3d58))

* no-curation, store processed data in user-specified `processed_data_dir` if provided ([`4397dd7`](https://github.com/datajoint/element-array-ephys/commit/4397dd7217b4595dc7fef498e6db456373ce50df))

* helper for `ProbeInsertion` - `auto_generate_entries(session_key)` ([`de84ce0`](https://github.com/datajoint/element-array-ephys/commit/de84ce0529c13eed5d1c1199062663f6a3888af2))

* improve kilosort loading routine - add `validate()` method ([`b7c0845`](https://github.com/datajoint/element-array-ephys/commit/b7c0845bc0514f3f435d79f6ba6fff86693b166a))

* minor bug fix ([`adfad95`](https://github.com/datajoint/element-array-ephys/commit/adfad9528d17689714d11b1ac1710d6f1a74756a))

* make `clustering_output_dir` user-input optional, auto infer ([`590310e`](https://github.com/datajoint/element-array-ephys/commit/590310ea8fc0829e60fd1113d99e83e75c78142d))

* remove `Curation` ([`a39a9b1`](https://github.com/datajoint/element-array-ephys/commit/a39a9b1b456c5b6fd49c2faa43f5d551e5f7901c))

* copied `ephys` to `ephys-no-curation`, added `recording_duration`, make ([`042cc46`](https://github.com/datajoint/element-array-ephys/commit/042cc460f48429e3b7c20eb01d49861d01357192))

* Update README ([`cdb9182`](https://github.com/datajoint/element-array-ephys/commit/cdb9182880dcca7f1070a2e2554d513488911cb2))

* Populated .md files ([`d8fca5b`](https://github.com/datajoint/element-array-ephys/commit/d8fca5bac61e8aacd3a24cffe8c6641e76512a05))

* Merge pull request #96 from ttngu207/main

bugfix - remove % in attributes&#39; comments AND add QC to `ephys_precluster` ([`4a6bc31`](https://github.com/datajoint/element-array-ephys/commit/4a6bc31e026de1869292e018efc5a817d01969e5))

* add QC to `ephys_precluster` ([`e21302f`](https://github.com/datajoint/element-array-ephys/commit/e21302f5a4c80b27aa4e66bf51e99f33153c4ebf))

* bugfix - remove `%` in attributes&#39; comments ([`57a1c1d`](https://github.com/datajoint/element-array-ephys/commit/57a1c1d7d067f683e321656f669ad1c14be25fbe))

* Merge pull request #87 from ttngu207/main

QC metrics ([`54c8413`](https://github.com/datajoint/element-array-ephys/commit/54c84137bbe55841b0e0db6079e0259d5240390f))

* Update CHANGELOG.md

Co-authored-by: Kabilar Gunalan &lt;kabilar@datajoint.com&gt; ([`7ce35ab`](https://github.com/datajoint/element-array-ephys/commit/7ce35ab8f47de621d5c0df3831c86c050a13b886))

* Apply suggestions from code review

Co-authored-by: Dimitri Yatsenko &lt;dimitri@datajoint.com&gt; ([`100cf84`](https://github.com/datajoint/element-array-ephys/commit/100cf84cbb5a8e59ae98b57b63aa7ec789669a34))

* apply PR review&#39;s suggestions ([`0a94aa9`](https://github.com/datajoint/element-array-ephys/commit/0a94aa9b2519e8168b297bcb7f050ed3894774e3))

* Merge branch &#39;datajoint:main&#39; into main ([`b126801`](https://github.com/datajoint/element-array-ephys/commit/b126801d6bc876465e47fcb10f7fc19aeb215183))

* bump version, add to CHANGELOG ([`c250857`](https://github.com/datajoint/element-array-ephys/commit/c25085792607a23021607d5c35bbde7d2890a32f))

* Apply suggestions from code review ([`35e8193`](https://github.com/datajoint/element-array-ephys/commit/35e8193fbc3d199fbf99932623bbafaa371fca12))

* code cleanup ([`7f948f7`](https://github.com/datajoint/element-array-ephys/commit/7f948f70e19b13f32d9e6ce6c178f34d6a5db665))

* Merge branch &#39;main&#39; of https://github.com/ttngu207/element-array-ephys ([`65554dc`](https://github.com/datajoint/element-array-ephys/commit/65554dc51486b1154c0f9bb844e6dfe18e2774cd))

* Merge branch &#39;datajoint:main&#39; into main ([`d884f01`](https://github.com/datajoint/element-array-ephys/commit/d884f01349358c82a9cc15414b7ba660f9fa2bad))

* add QC metrics ([`1773e23`](https://github.com/datajoint/element-array-ephys/commit/1773e23329852c96c8ba92c96144ea07e2b07036))

* Merge branch &#39;main&#39; of https://github.com/datajoint/element-array-ephys into main ([`4ab2a6c`](https://github.com/datajoint/element-array-ephys/commit/4ab2a6c9544e653efc50380c6117ca8f30facc5e))

* Merge branch &#39;main&#39; of https://github.com/ttngu207/element-array-ephys into main ([`b29b6b0`](https://github.com/datajoint/element-array-ephys/commit/b29b6b0ba0f715e00b378fbaf25e0af4e94373aa))

* Merge pull request #95 from guzman-raphael/docs

Documentation revamp ([`efedd37`](https://github.com/datajoint/element-array-ephys/commit/efedd37c5bb4798ad7ba7975d0506490e440f9f6))

* Update with recent styling updates. ([`c73c7b5`](https://github.com/datajoint/element-array-ephys/commit/c73c7b59f3ad1103673e7e30b0becaa49f7c3a9e))

* Remove unneeded comment. ([`0a8a67c`](https://github.com/datajoint/element-array-ephys/commit/0a8a67cf9c44b54c06d16293d1f64c5da755d1a0))

* Upgrade documentation to latest design and automation. ([`a304d31`](https://github.com/datajoint/element-array-ephys/commit/a304d316d3fb0df4e186c68b9a20ffcfd4e80457))

* Create u24_element_release_call.yml ([`3b98067`](https://github.com/datajoint/element-array-ephys/commit/3b98067cbdd4a617f6c7de4b21ed475d953866dc))

* Create u24_element_tag_to_release.yml ([`ddd6a18`](https://github.com/datajoint/element-array-ephys/commit/ddd6a189bc787dcf4e8174cb10e1e762ff912598))

* Create u24_element_before_release.yml ([`ac9035d`](https://github.com/datajoint/element-array-ephys/commit/ac9035dfb6dea9d24aecf11487794b00ca0c8b12))

* Merge pull request #84 from A-Baji/documentation

Create documentation template ([`22c95a8`](https://github.com/datajoint/element-array-ephys/commit/22c95a8dee0ac264ed990d8a785b8b1e25e4ef41))

* code review updates ([`87b4e40`](https://github.com/datajoint/element-array-ephys/commit/87b4e40c839f8e888d2c31a789c4297289ac4582))

* update deploy-docs wf condition ([`9ff6089`](https://github.com/datajoint/element-array-ephys/commit/9ff6089e361156cec3184709c5d72fee4582efc2))

* enable wf ([`9b56f01`](https://github.com/datajoint/element-array-ephys/commit/9b56f0144d0c1d9035e2be94cd52e24e11bab477))

* fix logos ([`4a61ff2`](https://github.com/datajoint/element-array-ephys/commit/4a61ff294ace685edc71618cf5c0bf034ad11e4d))

* enable wf ([`6e0b460`](https://github.com/datajoint/element-array-ephys/commit/6e0b460055d60bad9c68a649f07ee928d54d5dee))

* disable wf ([`25c4876`](https://github.com/datajoint/element-array-ephys/commit/25c4876a6ee0205fd6387eaeff88e71d28e23ded))

* add permalin, fix mobile color, update logo ([`9ef29f2`](https://github.com/datajoint/element-array-ephys/commit/9ef29f26bfa8687d2f3c651e1368d50c1b27212f))

* github actions ([`3e3a943`](https://github.com/datajoint/element-array-ephys/commit/3e3a94353219989be282f70b68c22c53af4481b6))

* fix landing page nav link ([`fdbe144`](https://github.com/datajoint/element-array-ephys/commit/fdbe1441235894a4b21697555004c2b577af0ce1))

* open link in new tab example ([`d8a1607`](https://github.com/datajoint/element-array-ephys/commit/d8a1607fba5c3244185db51cb593314706f75eac))

* update override ([`7cfd746`](https://github.com/datajoint/element-array-ephys/commit/7cfd74643f751d38525f77d17524fd5664a85e10))

* fix navigation override ([`88c5ed1`](https://github.com/datajoint/element-array-ephys/commit/88c5ed1037e6d5a156e038383975e68d0ec8d18b))

* cleanup navigation layout ([`d0e7abb`](https://github.com/datajoint/element-array-ephys/commit/d0e7abbfe476aa45edecdb0c50573274ab44d0c6))

* add datajoint social link ([`052e589`](https://github.com/datajoint/element-array-ephys/commit/052e589ee403e0435b53d32f3c4f498b736f191c))

* change light mode footer color ([`111fcf5`](https://github.com/datajoint/element-array-ephys/commit/111fcf51899c4ee6602e308067a6a9e3f566d48b))

* re enable wfs ([`3c2226d`](https://github.com/datajoint/element-array-ephys/commit/3c2226d89f8aff13c0c5f0bef196944b80694ef0))

* disable wfs ([`7def7f3`](https://github.com/datajoint/element-array-ephys/commit/7def7f3364d1472a09fd9bec74a74eb2b7758d89))

* change dark theme ([`2dca304`](https://github.com/datajoint/element-array-ephys/commit/2dca304aacf1ba89f464703c3fed390438ad74fb))

* dark mode tweak ([`26a2818`](https://github.com/datajoint/element-array-ephys/commit/26a2818e4885750928761a2d2db0884e5bdad4ab))

* update source block dark mode color ([`0c889e2`](https://github.com/datajoint/element-array-ephys/commit/0c889e2e8a9183ce48cad4c22654239d92e7dba4))

* tweak docstring example ([`a0e223a`](https://github.com/datajoint/element-array-ephys/commit/a0e223aec75c626d05786657b4f2b45b1b31a9d9))

* add social links to footer ([`656242e`](https://github.com/datajoint/element-array-ephys/commit/656242ebddfaccae726d664fa2b4e4c7b6603f0f))

* landing page nav link ([`b5b8bab`](https://github.com/datajoint/element-array-ephys/commit/b5b8bab6d362e37d796dfb5531a65d9cb35b0eb6))

* re enable wf ([`109965b`](https://github.com/datajoint/element-array-ephys/commit/109965b504047147203647815b1c87f07d043c62))

* disable other wf ([`a7b3d11`](https://github.com/datajoint/element-array-ephys/commit/a7b3d116c969875c20adcd1e33142f34e67f133e))

* apply suggestions from review ([`9f5753c`](https://github.com/datajoint/element-array-ephys/commit/9f5753c1deba2ea7ac21cdfb08be63031b2d258b))

* disable mike install ([`d1ee89b`](https://github.com/datajoint/element-array-ephys/commit/d1ee89b8e097a521d45647d7413c1c635b1ed6ae))

* re enable other gh action wf ([`9b70fa1`](https://github.com/datajoint/element-array-ephys/commit/9b70fa1d7103b340dda95d9b3b21da7c8a540679))

* disable other gh action wf ([`f1b1868`](https://github.com/datajoint/element-array-ephys/commit/f1b186835cb0bf1a7e98f731efb3ff42d336c4b0))

* move docker files to docs/ ([`33d6fcc`](https://github.com/datajoint/element-array-ephys/commit/33d6fcc39b52e7da1e698945fbd991bf8660f434))

* comment cleanup ([`5a936af`](https://github.com/datajoint/element-array-ephys/commit/5a936afd88b006948fb89cc39e625429139e3db2))

* add mike workflow example ([`03b6dc3`](https://github.com/datajoint/element-array-ephys/commit/03b6dc395c8168aac8f616daf59971c38872685d))

* add mike for future use ([`6f7eedf`](https://github.com/datajoint/element-array-ephys/commit/6f7eedf4ca16b065e6e7b2e3ec6c061ba6b8608e))

* re enable other wf jobs ([`9bdb418`](https://github.com/datajoint/element-array-ephys/commit/9bdb4187d8a9c72f8c414147a37875200d72b2cc))

* add missing dependencies ([`5c44e80`](https://github.com/datajoint/element-array-ephys/commit/5c44e8018882d66b7f73c32ecac7fe6202737ed0))

* add config file path ([`00d3ca2`](https://github.com/datajoint/element-array-ephys/commit/00d3ca22e2fa6634fb62ac94c5958c233a218528))

* disable other jobs ([`5469e77`](https://github.com/datajoint/element-array-ephys/commit/5469e77a48483c4870c42721c89802d536044ef7))

* docker and github wf ([`730614b`](https://github.com/datajoint/element-array-ephys/commit/730614b8feb1b966e2150813ad9339f13a1a9bff))

* small change ([`f40c188`](https://github.com/datajoint/element-array-ephys/commit/f40c1887bbcc1b6e1111258d35659709eb7f4cc8))

* move docs to src ([`33224a9`](https://github.com/datajoint/element-array-ephys/commit/33224a9087ad988fb0b4429deca4d385017d3043))

* cleanup ([`4399034`](https://github.com/datajoint/element-array-ephys/commit/4399034e9ae66ae2cbcd6df998f0c65466147c16))

* clean up and tweak dark mode theme ([`72e3aa6`](https://github.com/datajoint/element-array-ephys/commit/72e3aa6c29646351a602908696b84b7b852d31d7))

* tweak dark mode theme for codeblocks ([`95e3925`](https://github.com/datajoint/element-array-ephys/commit/95e3925d85a1d1da5122bba160c89afe6c70f49c))

* docstring example ([`c4d3bde`](https://github.com/datajoint/element-array-ephys/commit/c4d3bde2fb9c4e4bb5f9981468c360808428fe0a))

* light and dark themes ([`b1f7399`](https://github.com/datajoint/element-array-ephys/commit/b1f7399984bc3f3c1707b43723c342ca6b8cd42e))

* dj light theme ([`724d870`](https://github.com/datajoint/element-array-ephys/commit/724d870489c9346044bc305065622dee04a05f0e))

* set up mkdocs ([`f2a5e7c`](https://github.com/datajoint/element-array-ephys/commit/f2a5e7c0bcd7c85f83a535a53a739cda4e81e026))

## v0.1.4 (2022-07-11)

### Unknown

* Merge pull request #83 from kabilar/main

Fix for `spike_depths` attribute ([`ee0e179`](https://github.com/datajoint/element-array-ephys/commit/ee0e179d0ed02212f03a36382a07409b6ba2f823))

* Update changelog ([`a97dd3c`](https://github.com/datajoint/element-array-ephys/commit/a97dd3c4fd468832721d6bedc3b796a89f01b3b9))

* Fix if statement ([`c66ff8f`](https://github.com/datajoint/element-array-ephys/commit/c66ff8f7311768f6916b859319570a4d267a423f))

* Update changelog ([`0da5e91`](https://github.com/datajoint/element-array-ephys/commit/0da5e915a97f6c9b1ed5d5f8c8b7c551def38440))

* Update changelog and version ([`1865be6`](https://github.com/datajoint/element-array-ephys/commit/1865be641b4a62b87acb8b5a68c0ceb8914aede8))

* Fix for truth value of array ([`787d33d`](https://github.com/datajoint/element-array-ephys/commit/787d33d6ce478976a2c2e49d72fe11be90b5782f))

## v0.1.3 (2022-06-16)

### Unknown

* Merge pull request #79 from kabilar/main

Update `precluster_output_dir` to nullable ([`ecd6a4c`](https://github.com/datajoint/element-array-ephys/commit/ecd6a4c0212ebb54dd1d256a384aa0b8bf7785f7))

* Set precluster_output_dir to nullable ([`90f3ed1`](https://github.com/datajoint/element-array-ephys/commit/90f3ed177587dc364e9b2548afb515809b549ec8))

## v0.1.2 (2022-06-09)

### Unknown

* Merge pull request #78 from kabilar/main

Fix for case where `pc_features.npy` does not exist ([`a01530c`](https://github.com/datajoint/element-array-ephys/commit/a01530ca2216787f2b69906f596a4b785323cf50))

* Fix format ([`6b6f448`](https://github.com/datajoint/element-array-ephys/commit/6b6f448c9b3ca88d3106b37bb5a7bb474ce4d157))

* Update element_array_ephys/ephys_chronic.py ([`558e0b9`](https://github.com/datajoint/element-array-ephys/commit/558e0b94537e0d3b3c9c3d83823e9ff8a9212c57))

* Update element_array_ephys/ephys_acute.py ([`44dbe8c`](https://github.com/datajoint/element-array-ephys/commit/44dbe8cc84bf009bee1abdf22adc118ee6564457))

* Update element_array_ephys/readers/kilosort.py

Co-authored-by: Thinh Nguyen &lt;thinh@vathes.com&gt; ([`a392e57`](https://github.com/datajoint/element-array-ephys/commit/a392e57868ec9d9b356cf3c1a6e57b0dc33fbb1b))

* Update element_array_ephys/ephys_precluster.py

Co-authored-by: Thinh Nguyen &lt;thinh@vathes.com&gt; ([`b3922fc`](https://github.com/datajoint/element-array-ephys/commit/b3922fc58e213b52ca0481c45ca0bcc7a01d1e1c))

* Update version and changelog ([`3a2671a`](https://github.com/datajoint/element-array-ephys/commit/3a2671a1b4d4dff344ac3431357482a4ce5c270c))

* Handle case where pc_features does not exist ([`c16fda2`](https://github.com/datajoint/element-array-ephys/commit/c16fda209410974116eea0bc893eb8542ca2afa0))

* Flatten channel map ([`cdce624`](https://github.com/datajoint/element-array-ephys/commit/cdce624300b20272d5662f0fdb7ec20d436148e1))

* Handle case where pc_features does not exist ([`c428e47`](https://github.com/datajoint/element-array-ephys/commit/c428e47c17a69fd0812cf4ad224db0ccff0ca036))

* Merge branch &#39;main&#39; of https://github.com/datajoint/element-array-ephys into main ([`d53f7a9`](https://github.com/datajoint/element-array-ephys/commit/d53f7a9228dc03c86c0ccb8392f39bc8d67d3b40))

## v0.1.1 (2022-06-01)

### Unknown

* Merge pull request #72 from kabilar/main

Add attributes to describe parameter list ([`a20ab9b`](https://github.com/datajoint/element-array-ephys/commit/a20ab9b7879e8cf4131206bf3dbb099d557b0233))

* Merge branch &#39;main&#39; of https://github.com/kabilar/element-array-ephys into main ([`d618c55`](https://github.com/datajoint/element-array-ephys/commit/d618c5577f951791cc646ba6617847058b65516f))

* Update CHANGELOG.md ([`81e1643`](https://github.com/datajoint/element-array-ephys/commit/81e164399caf1ee14141890a02c904d23530d6d5))

* Update element_array_ephys/ephys_precluster.py ([`34a544e`](https://github.com/datajoint/element-array-ephys/commit/34a544e08509edf46962b0cf0c3753477100e732))

* Set spike_depths as nullable attribute ([`4142468`](https://github.com/datajoint/element-array-ephys/commit/4142468ba4f83fd95da6a6f988e95f97b5907555))

* Update length ([`2ce12c1`](https://github.com/datajoint/element-array-ephys/commit/2ce12c1cfed6e4e983dd19cfcb5f0d021f49c845))

* Update diff ([`8366b60`](https://github.com/datajoint/element-array-ephys/commit/8366b60c641b15af5aa40720248564fd606e4bd5))

* Update version and changelog ([`ad9a4b9`](https://github.com/datajoint/element-array-ephys/commit/ad9a4b97d741a72fc18c093c5806f9a732dac54b))

* Add description attribute ([`08fb06a`](https://github.com/datajoint/element-array-ephys/commit/08fb06af3ad45be1c8dcf1576303d3c085f8593e))

* Merge pull request #65 from kabilar/main

Add `ephys_precluster` module ([`3eeae51`](https://github.com/datajoint/element-array-ephys/commit/3eeae51bd34570c95dcec945eb1b55771edeb902))

* Add ephys_chronic image ([`c82c23b`](https://github.com/datajoint/element-array-ephys/commit/c82c23b72e169b0bb5b2cc3057b5af6323e052d6))

* Add precluster image ([`a31abba`](https://github.com/datajoint/element-array-ephys/commit/a31abba2ef5f5bf8d49449dda4a20286765d8a87))

* Raise error ([`92e30ee`](https://github.com/datajoint/element-array-ephys/commit/92e30ee03194ea5cf1eeba076c324f73c9b7ebf6))

* Merge branch &#39;main&#39; of kabilar/element-array-ephys ([`cd31e0b`](https://github.com/datajoint/element-array-ephys/commit/cd31e0b292ffda20cc0d0f6096591fc7e1350329))

* Update element_array_ephys/ephys_precluster.py ([`5bbb727`](https://github.com/datajoint/element-array-ephys/commit/5bbb727d4ccccdab1add41c21ad505fddde94ebe))

* Update name ([`3df0981`](https://github.com/datajoint/element-array-ephys/commit/3df0981f7011d6871bf3772f4ba98917e89cf80b))

* Update changelog ([`bbe9f3f`](https://github.com/datajoint/element-array-ephys/commit/bbe9f3f445efb35f287405469b6226d0bd4a2f7e))

* Update version ([`44c86bf`](https://github.com/datajoint/element-array-ephys/commit/44c86bfdf665f71946375775d5b1ac12323b08d5))

* Merge branch &#39;main&#39; of https://github.com/kabilar/element-array-ephys into main ([`2bd2234`](https://github.com/datajoint/element-array-ephys/commit/2bd2234030f477205108054ec70dc195e5ebae8c))

* Update element_array_ephys/ephys_precluster.py ([`dc0fc1f`](https://github.com/datajoint/element-array-ephys/commit/dc0fc1f50aa1d950d89cb8b355e992a3fdcb3125))

* Update element_array_ephys/ephys_precluster.py ([`f2baf12`](https://github.com/datajoint/element-array-ephys/commit/f2baf12fe2d9a6417f2d63674563570bb72453af))

* Update element_array_ephys/ephys_precluster.py ([`ec0ebf2`](https://github.com/datajoint/element-array-ephys/commit/ec0ebf206dcfff530b0d9f3c2c7dfa50ef1d66f3))

* Update element_array_ephys/ephys_precluster.py ([`8d793ac`](https://github.com/datajoint/element-array-ephys/commit/8d793ac82d3be366d4a07900c399d39210ff2ad0))

* Add documentation for ephys modules ([`644a114`](https://github.com/datajoint/element-array-ephys/commit/644a114b72e8ae4efc332291721d3987aa22a007))

* Rename image ([`91950b0`](https://github.com/datajoint/element-array-ephys/commit/91950b0b1d88d92563ec5ee02245602e0c022480))

* Merge &#39;main&#39; of datajoint/element-array-ephys ([`1b60995`](https://github.com/datajoint/element-array-ephys/commit/1b60995453afae074d2b32d33dabbf06044b1dad))

* Merge pull request #44 from bendichter/convert_to_nwb

Convert to nwb ([`7a4fba9`](https://github.com/datajoint/element-array-ephys/commit/7a4fba9ba51d6ee1cf21bb7eaf87a59b4accfd44))

* Update element_array_ephys/export/nwb/nwb.py

Co-authored-by: Dimitri Yatsenko &lt;dimitri@datajoint.com&gt; ([`9ee6088`](https://github.com/datajoint/element-array-ephys/commit/9ee60885e638e85746a093c791a848dbd37f2472))

* Update element_array_ephys/export/nwb/nwb.py

Co-authored-by: Dimitri Yatsenko &lt;dimitri@datajoint.com&gt; ([`441cfe2`](https://github.com/datajoint/element-array-ephys/commit/441cfe2e2ab00765ad9603d28f0bd8a50d48d1d1))

* Merge remote-tracking branch &#39;origin/convert_to_nwb&#39; into convert_to_nwb ([`6fc51b0`](https://github.com/datajoint/element-array-ephys/commit/6fc51b055ef77069f04bc72cf7999d8e0c6717b0))

* Update element_array_ephys/export/nwb/nwb.py

Co-authored-by: Kabilar Gunalan &lt;kabilar@datajoint.com&gt; ([`9f9872c`](https://github.com/datajoint/element-array-ephys/commit/9f9872c37eb325b441703d4204889f6298d1ba4e))

* Update element_array_ephys/export/nwb/nwb.py

Co-authored-by: Kabilar Gunalan &lt;kabilar@datajoint.com&gt; ([`fb98327`](https://github.com/datajoint/element-array-ephys/commit/fb983274e8294fbee5703e6ccaaa3d46ad1394b4))

* Update element_array_ephys/export/nwb/nwb.py

Co-authored-by: Kabilar Gunalan &lt;kabilar@datajoint.com&gt; ([`d94453b`](https://github.com/datajoint/element-array-ephys/commit/d94453b1e7704ef41f830ae4c7e06f569a37f545))

* remove ephys_no_curation.py ([`3e07c61`](https://github.com/datajoint/element-array-ephys/commit/3e07c61b7556fffea3c7daa7409f70f51541e76e))

* Update element_array_ephys/export/nwb/nwb.py

Co-authored-by: Kabilar Gunalan &lt;kabilar@datajoint.com&gt; ([`59028cb`](https://github.com/datajoint/element-array-ephys/commit/59028cb22c1ee533d4635d677eea897a933cbf71))

* Update element_array_ephys/export/nwb/nwb.py

Co-authored-by: Kabilar Gunalan &lt;kabilar@datajoint.com&gt; ([`068ea3d`](https://github.com/datajoint/element-array-ephys/commit/068ea3d3682cc00d988807e804744102f8c0e359))

* trying clustering_query..proj() ([`c2004ea`](https://github.com/datajoint/element-array-ephys/commit/c2004eabf06822d563e06515e50476ac76ec3610))

* rmv units_query ([`0152c5d`](https://github.com/datajoint/element-array-ephys/commit/0152c5d97dd6b40119312f50da62f31e476dbad4))

* fix insertion record ([`82c8655`](https://github.com/datajoint/element-array-ephys/commit/82c86559e86e64580b740325741a5f62e6cf037f))

* add explanation for index parameter ([`707adff`](https://github.com/datajoint/element-array-ephys/commit/707adff4e6fbb451e0582115b85b72b36002ba9e))

* Merge remote-tracking branch &#39;origin/convert_to_nwb&#39; into convert_to_nwb ([`bc54009`](https://github.com/datajoint/element-array-ephys/commit/bc54009c7520d29d13b2a5c5f3def3e8888502c6))

* Update element_array_ephys/export/nwb/nwb.py

Co-authored-by: Kabilar Gunalan &lt;kabilar@datajoint.com&gt; ([`842beec`](https://github.com/datajoint/element-array-ephys/commit/842beec80ed5d846aed29a48575db1de1457bdf9))

* Update element_array_ephys/export/nwb/nwb.py

Co-authored-by: Kabilar Gunalan &lt;kabilar@datajoint.com&gt; ([`766f4eb`](https://github.com/datajoint/element-array-ephys/commit/766f4eb0962cf36e10290cc1ff4dfb27ad74de87))

* Update element_array_ephys/export/nwb/nwb.py

Co-authored-by: Kabilar Gunalan &lt;kabilar@datajoint.com&gt; ([`2526812`](https://github.com/datajoint/element-array-ephys/commit/2526812f664e7e58a9db368e293dc8e7927615d7))

* Update element_array_ephys/export/nwb/nwb.py

Co-authored-by: Kabilar Gunalan &lt;kabilar@datajoint.com&gt; ([`15044cf`](https://github.com/datajoint/element-array-ephys/commit/15044cf4c85ae67f9e2a73031980180380a0d974))

* Update element_array_ephys/export/nwb/nwb.py

Co-authored-by: Kabilar Gunalan &lt;kabilar@datajoint.com&gt; ([`2e621c0`](https://github.com/datajoint/element-array-ephys/commit/2e621c02a4809e1c5a540ddc7e22c07f5fcdec1b))

* Update element_array_ephys/export/nwb/nwb.py

Co-authored-by: Kabilar Gunalan &lt;kabilar@datajoint.com&gt; ([`a01ee9c`](https://github.com/datajoint/element-array-ephys/commit/a01ee9ca8e277265382adb674b707ba67f173c01))

* Update element_array_ephys/export/nwb/nwb.py

Co-authored-by: Kabilar Gunalan &lt;kabilar@datajoint.com&gt; ([`cbefcde`](https://github.com/datajoint/element-array-ephys/commit/cbefcde9f81de6be64eeb31a04631fbb00fe6431))

* add explanation in docstring of add_ephys_units_to_nwb ([`3971fe6`](https://github.com/datajoint/element-array-ephys/commit/3971fe6632030d789ab6b159936d3f4dc2f5f878))

* Merge remote-tracking branch &#39;origin/convert_to_nwb&#39; into convert_to_nwb ([`a7b2abb`](https://github.com/datajoint/element-array-ephys/commit/a7b2abb99baf7295e4549ba5eb0edf7afb9acb63))

* Update element_array_ephys/export/nwb/nwb.py ([`c200699`](https://github.com/datajoint/element-array-ephys/commit/c200699f35d8f90be07f4a8fb194e33b504d9f78))

* Update element_array_ephys/export/nwb/nwb.py

Co-authored-by: Kabilar Gunalan &lt;kabilar@datajoint.com&gt; ([`2ee2bd5`](https://github.com/datajoint/element-array-ephys/commit/2ee2bd544ed777c537dc18b6421b4764462f3482))

* Update element_array_ephys/export/nwb/nwb.py

Co-authored-by: Chris Brozdowski &lt;CBrozdowski@yahoo.com&gt; ([`9e72773`](https://github.com/datajoint/element-array-ephys/commit/9e72773d701550b9c613dfb208f6543ce227e803))

* Merge branch &#39;main&#39; into convert_to_nwb ([`b3779e5`](https://github.com/datajoint/element-array-ephys/commit/b3779e58a0b5f2be73abfc07c5425272089a9250))

* fix docstring for get_electrodes_mapping ([`acdb5f9`](https://github.com/datajoint/element-array-ephys/commit/acdb5f9d25c4f1c1f6e89dd37fd8e1697327a8e9))

* Merge remote-tracking branch &#39;origin/convert_to_nwb&#39; into convert_to_nwb ([`826335b`](https://github.com/datajoint/element-array-ephys/commit/826335be00a09481b17b06ce6901c152490f301e))

* Update element_array_ephys/export/nwb/nwb.py

Co-authored-by: Kabilar Gunalan &lt;kabilar@datajoint.com&gt; ([`49fac08`](https://github.com/datajoint/element-array-ephys/commit/49fac083d5de5c853acb8833460edaa07132638e))

* Update element_array_ephys/export/nwb/nwb.py

Co-authored-by: Kabilar Gunalan &lt;kabilar@datajoint.com&gt; ([`ba6cbcf`](https://github.com/datajoint/element-array-ephys/commit/ba6cbcf1c3f03cd6a68b96a5b10e28398d9d60e9))

* specify releases for dependencies ([`daccfc4`](https://github.com/datajoint/element-array-ephys/commit/daccfc4dd7a48142739dfee5b00a3ee7c9624d19))

* add docstring for gain_helper ([`12974ff`](https://github.com/datajoint/element-array-ephys/commit/12974ff0df04fb8dc15b650e5cbbd3b51aa6340f))

* Update element_array_ephys/export/nwb/README.md

Co-authored-by: Kabilar Gunalan &lt;kabilar@datajoint.com&gt; ([`00c3691`](https://github.com/datajoint/element-array-ephys/commit/00c369144a50cc34acb0fdab12ea0ac93b6627a4))

* Update element_array_ephys/export/nwb/nwb.py

Co-authored-by: Kabilar Gunalan &lt;kabilar@datajoint.com&gt; ([`abccafa`](https://github.com/datajoint/element-array-ephys/commit/abccafa522f930f099eefd49c14e69eb95c56067))

* Update element_array_ephys/export/nwb/README.md

Co-authored-by: Kabilar Gunalan &lt;kabilar@datajoint.com&gt; ([`24ac2c0`](https://github.com/datajoint/element-array-ephys/commit/24ac2c038b3e7403a3130e43a385c1bab0acb8f5))

* Update element_array_ephys/export/nwb/nwb.py

Co-authored-by: Kabilar Gunalan &lt;kabilar@datajoint.com&gt; ([`a1fb193`](https://github.com/datajoint/element-array-ephys/commit/a1fb1934cb34b1ebedb11a10f514cb1ce24b9e00))

* Update element_array_ephys/export/nwb/README.md

Co-authored-by: Kabilar Gunalan &lt;kabilar@datajoint.com&gt; ([`584c738`](https://github.com/datajoint/element-array-ephys/commit/584c738982b53d5988e338302e72f7967b43fe2d))

* Update element_array_ephys/export/nwb/nwb.py

Co-authored-by: Kabilar Gunalan &lt;kabilar@datajoint.com&gt; ([`c68f2ca`](https://github.com/datajoint/element-array-ephys/commit/c68f2cabfe9d362d4a176a80eb8b87f068f8e317))

* Update element_array_ephys/export/nwb/nwb.py

Co-authored-by: Chris Brozdowski &lt;CBrozdowski@yahoo.com&gt; ([`3ae6e2d`](https://github.com/datajoint/element-array-ephys/commit/3ae6e2d0f96ac74d77cf5617719abc638c1fba66))

* Update element_array_ephys/ephys_acute.py

Co-authored-by: Chris Brozdowski &lt;CBrozdowski@yahoo.com&gt; ([`e973743`](https://github.com/datajoint/element-array-ephys/commit/e9737435f55f8b3ef48c10ee691c0bfd37dd7e21))

* Update element_array_ephys/export/nwb/nwb.py

Co-authored-by: Kabilar Gunalan &lt;kabilar@datajoint.com&gt; ([`b373b26`](https://github.com/datajoint/element-array-ephys/commit/b373b264080e36e379e775499a1b3166c004b546))

* fix imports ([`cc14671`](https://github.com/datajoint/element-array-ephys/commit/cc146716036cf8dba74a241160f0b67ae2c552bd))

* rmv tests (they are moved to the ephys workflow) ([`c63c8c4`](https://github.com/datajoint/element-array-ephys/commit/c63c8c42fc640319d4cfc4acf90e1b1595e63e3b))

* Merge remote-tracking branch &#39;cbroz1/ben&#39; into convert_to_nwb

# Conflicts:
#	element_array_ephys/export/nwb/nwb.py ([`5b942b3`](https://github.com/datajoint/element-array-ephys/commit/5b942b33d1e4611367f962c70ec571b3d99f2143))

* adjust imports in __init__ and nwb.py ([`782a4a5`](https://github.com/datajoint/element-array-ephys/commit/782a4a5ee0cb7c6e1e382c2985dce45c418c7c19))

* Merge pull request #2 from ttngu207/chris-nwb

import the correct ephys module that has been activated ([`2beb41b`](https://github.com/datajoint/element-array-ephys/commit/2beb41b9af183468846a0c03b7de8ab740f855c4))

* import the correct ephys module that has been activated ([`b4ffe1d`](https://github.com/datajoint/element-array-ephys/commit/b4ffe1d68fe068a5bd233ad8c9adb3bda09cf145))

* Merge pull request #1 from ttngu207/chris-nwb

nwb function specification in linking_module ([`6d7ad7c`](https://github.com/datajoint/element-array-ephys/commit/6d7ad7c131037b882c1cf7a1183b5438e4635dbd))

* nwb function specification in linking_module ([`bf5e82a`](https://github.com/datajoint/element-array-ephys/commit/bf5e82aa63d771151ddd7348d5d37832cad8ac9d))

* Avoid linking_module issues. See details.

- add __init__ + schema from git/ttngu207/element_array_ephys@no-curation
- add ephys_no_curation schema to match the schema ben has been pulling from
   - how should we address this in the element? given not currently default
- remove unused imports
- import datajoint and element_interface.utils find_full_path
- add arguments to main export function:
   - schema names as datajoint config database prefix default
   - ephys_root_data_dir - default to dj.config or none
- add create_virtual_module statements to avoid activate(schema,linking_module=unknown)
- declare ephys and probe as global
- add assert errors for ephys_root_data_dir!=None when needed
- pass ephys_root_data_dir to relevant functions
- above permits: from element_array_ephys.export.nwb.nwb import ecephys_session_to_nwb ([`2595dae`](https://github.com/datajoint/element-array-ephys/commit/2595daee613f1b080b6bc4a6743865e8f9b42dc7))

* Rebase, squashed. See Details

Add element_data_loader for multiple root dirs
Update author
Fix import
Fix OpenEphys session path
Update directory path
Add print statement
Fix for missing `fileTimeSecs`
Update error message
Suggested adds re upstream components
Directing to workflow for upstream `SkullReference` and utility functions ([`a557b17`](https://github.com/datajoint/element-array-ephys/commit/a557b17caaa0d779f91bee105a17fb1418121e00))

* rmv subject_id ([`806684f`](https://github.com/datajoint/element-array-ephys/commit/806684f4534c11982e78d50264f47257c3c3018f))

* import GenericDataChunkIterator from hdmf ([`a7f4624`](https://github.com/datajoint/element-array-ephys/commit/a7f46242d849a65149cb7207a87ec489402f7452))

* add tests for getting lfp data from datajoint ([`fafdde1`](https://github.com/datajoint/element-array-ephys/commit/fafdde1257aafd00b3c47739be8a73a4c7f09087))

* Merge remote-tracking branch &#39;origin/convert_to_nwb&#39; into convert_to_nwb ([`63b545d`](https://github.com/datajoint/element-array-ephys/commit/63b545dec4d788132bf28832d1156488d478e47e))

* Update element_array_ephys/export/nwb/nwb.py

Co-authored-by: Kabilar Gunalan &lt;kabilar@datajoint.com&gt; ([`9374d94`](https://github.com/datajoint/element-array-ephys/commit/9374d94d1372f6f964ecb3bd628c5f97c18c261b))

* Merge branch &#39;main&#39; into convert_to_nwb ([`f8027cc`](https://github.com/datajoint/element-array-ephys/commit/f8027ccedbae3418f3cd8e54d6a0197aeea41c65))

* update import path ([`6338daf`](https://github.com/datajoint/element-array-ephys/commit/6338dafd0a00c2eb862726977cf6c2470a8c7b1a))

* add tests ([`4cab8c8`](https://github.com/datajoint/element-array-ephys/commit/4cab8c86620094e912e7193c3a579989046d26b1))

* refactor into gains_helper ([`7953662`](https://github.com/datajoint/element-array-ephys/commit/79536621e8b6b9f35ed4cb61e5384c433da28bc0))

* correctly set conversion and channel_conversion ([`7deb00f`](https://github.com/datajoint/element-array-ephys/commit/7deb00f8e360c5b23e8b63333c55b50ece5334c3))

* ephys.find_full_path ([`654d567`](https://github.com/datajoint/element-array-ephys/commit/654d567f8701940bdec6c426d65cb619c3fa2016))

* import os ([`ba3f86a`](https://github.com/datajoint/element-array-ephys/commit/ba3f86a2a956e26beb7223cbf09dc779de894a8b))

* standardize slashes ([`8d3df71`](https://github.com/datajoint/element-array-ephys/commit/8d3df711fc6f79282e81ac2794c2ca0b5d19c10c))

* ephys.get_ephys_root_data_dir ([`e992478`](https://github.com/datajoint/element-array-ephys/commit/e99247874924a39596510fb144f3272f00886804))

* import probe ([`379ae11`](https://github.com/datajoint/element-array-ephys/commit/379ae11c718242d04cc6c2eb62a76a433d2b94cf))

* import session_to_nwb ([`1dccb68`](https://github.com/datajoint/element-array-ephys/commit/1dccb6893fa8a77408e598f278995998b5c8b45c))

* import from workflow pipeline ([`5de22e0`](https://github.com/datajoint/element-array-ephys/commit/5de22e06fa2db7b6e2005bbc094b05fc3581e850))

* import from workflow pipeline ([`1b67629`](https://github.com/datajoint/element-array-ephys/commit/1b676293227e03e77293c223f8d3f336c832ee59))

* fix nwbfile_kwargs logic ([`eb47ee5`](https://github.com/datajoint/element-array-ephys/commit/eb47ee506db1fd1929da2d557c17776c28675326))

* fix nwbfile_kwargs logic ([`96c57f7`](https://github.com/datajoint/element-array-ephys/commit/96c57f756243eb00ced7e157e417ab1af5881c10))

* add optional session keys to ecephys_session_to_nwb ([`365b43b`](https://github.com/datajoint/element-array-ephys/commit/365b43b0fa584a531b386afe74eb55e853773a2d))

* add datetime import ([`d1f3dab`](https://github.com/datajoint/element-array-ephys/commit/d1f3daba676385a54aa108bb554bce008594b96d))

* relative import of ephys ([`1363214`](https://github.com/datajoint/element-array-ephys/commit/13632147a56826fe7ccba065aa02e370950a72c0))

* refactor to include requirements for nwb conversion ([`013ae7d`](https://github.com/datajoint/element-array-ephys/commit/013ae7da1bc94ed2c6c5e87b79378e121826882b))

* add readme for exporting to NWB ([`19f78b7`](https://github.com/datajoint/element-array-ephys/commit/19f78b78ef4467f1edd7b3270a8df51cdf349a91))

* add some docstrings ([`b35ee72`](https://github.com/datajoint/element-array-ephys/commit/b35ee723dc39777dfede2334452021a5c234f6d8))

* add missing nwbfile arg ([`a7f846b`](https://github.com/datajoint/element-array-ephys/commit/a7f846be82b020c5b6ace34b524dd99c5438d345))

* Merge remote-tracking branch &#39;origin/convert_to_nwb&#39; into convert_to_nwb

# Conflicts:
#	element_array_ephys/export/nwb.py ([`b4d9c0e`](https://github.com/datajoint/element-array-ephys/commit/b4d9c0edd791f4f846b99c6bcc98ee688bbb508c))

* Update element_array_ephys/export/nwb.py

Co-authored-by: Dimitri Yatsenko &lt;dimitri@datajoint.com&gt; ([`e837452`](https://github.com/datajoint/element-array-ephys/commit/e83745247b09b369f6992590446c7167c32bddb7))

* Update element_array_ephys/export/nwb.py

Co-authored-by: Dimitri Yatsenko &lt;dimitri@datajoint.com&gt; ([`3298341`](https://github.com/datajoint/element-array-ephys/commit/3298341553063a8e2d165c25cfa18c0050047cde))

* Update element_array_ephys/export/nwb.py

Co-authored-by: Dimitri Yatsenko &lt;dimitri@datajoint.com&gt; ([`470b20a`](https://github.com/datajoint/element-array-ephys/commit/470b20aa3914fb8c0bed943c7d2f82c0df5dd4fd))

* * optimize imports
* black
* upgrade to latest version of conversion-tools
* upgrade to latest spikeinterface api
* ([`d924d57`](https://github.com/datajoint/element-array-ephys/commit/d924d57ea62ec24c791135a16f3ed397493084d5))

* add documentation ([`d2b93f2`](https://github.com/datajoint/element-array-ephys/commit/d2b93f2054c51848a20d1a8da44e3cc7a9586973))

* * add lfp from source
* docstrings
* json dump insertion into location
* ignore channel_conversion if all are 1
* black formatting ([`8d75a61`](https://github.com/datajoint/element-array-ephys/commit/8d75a61ac29b0d51381104472a8570db6c99bcb5))

* add draft convert to nwb ([`f3dd8d8`](https://github.com/datajoint/element-array-ephys/commit/f3dd8d80c0a0b9a4e16734a35cd0e2f4520a0142))

* Mult rootdirs. If sess_dir, check fullpath. Give OpenEphys fullpath. ([`720eb00`](https://github.com/datajoint/element-array-ephys/commit/720eb00e60c6df928f4e6cbe938d4db625feab58))

* Add Part table to track order of operations ([`3d8ec16`](https://github.com/datajoint/element-array-ephys/commit/3d8ec16a85507d4708ceedfdcffd4609546dc0f7))

* Merge branch &#39;main&#39; of https://github.com/kabilar/element-array-ephys into main ([`fee5b7c`](https://github.com/datajoint/element-array-ephys/commit/fee5b7ca39b564985ad5907dcbdfd58f719beeec))

* Merge pull request #1 from kabilar/precluster

Merge to main ([`1e92b04`](https://github.com/datajoint/element-array-ephys/commit/1e92b048c2da3dd50916df1825bf8e477de9bb05))

* Add pre-clustering tables to acute module ([`c7a155a`](https://github.com/datajoint/element-array-ephys/commit/c7a155a9d4ab384d667046fefb316df7d00ed656))

* Create copy of `ephys_acute.py` ([`122d9a2`](https://github.com/datajoint/element-array-ephys/commit/122d9a2a598263340d1dcdc49353ac3916ffc3b7))

* Merge pull request #64 from kabilar/main

Update README ([`14517a2`](https://github.com/datajoint/element-array-ephys/commit/14517a27288cf2a2ddfd526da64a2d32da69b156))

* Replace italics with back tick ([`b3f5b29`](https://github.com/datajoint/element-array-ephys/commit/b3f5b295ed233aedad5db1f872be18c2df644951))

* Replace italics with back tick ([`e8350d5`](https://github.com/datajoint/element-array-ephys/commit/e8350d5bd5e02a7150c7c5c2589daef8aaaa54a8))

* Add ephys schema text ([`112f325`](https://github.com/datajoint/element-array-ephys/commit/112f325d2bdc2e940756531282ac38f5e67bc67d))

* Add activation text ([`edc9d5d`](https://github.com/datajoint/element-array-ephys/commit/edc9d5dd3c972ea83042b3f844d5011d2602d1c3))

* Add ephys schema text ([`ab92d84`](https://github.com/datajoint/element-array-ephys/commit/ab92d847e2ec39d7f0d6dc851ac6ef26bfaf7bcb))

* Revert &#34;Create copy of `ephys_acute.py`&#34;

This reverts commit b66109b5e61297a10c1cc8a929115fa5955238e1. ([`000308f`](https://github.com/datajoint/element-array-ephys/commit/000308f7261794e6acd39762e524c27331bb1a0a))

* Add probe schema text ([`4f8699f`](https://github.com/datajoint/element-array-ephys/commit/4f8699fa5859e4a873c5dc32fbf4fae64e5073af))

* Create copy of `ephys_acute.py` ([`c7393fc`](https://github.com/datajoint/element-array-ephys/commit/c7393fcb4933846b6862548a47c29cadc4d97801))

* Revert &#34;Create copy of `ephys_acute.py`&#34;

This reverts commit b66109b5e61297a10c1cc8a929115fa5955238e1. ([`9ddfb4c`](https://github.com/datajoint/element-array-ephys/commit/9ddfb4c927cca7a8de40965752d7eac2e06bd07d))

* Update format ([`f940a71`](https://github.com/datajoint/element-array-ephys/commit/f940a719d9c409eb40b760f8513ba9737b5cf809))

* Add collapsible sections ([`338a796`](https://github.com/datajoint/element-array-ephys/commit/338a796ff2ca3cddf460309c6776034f6409aeed))

* Update format ([`9b68b03`](https://github.com/datajoint/element-array-ephys/commit/9b68b0332d1cd228327fea70459b1383b77b4473))

* Add collapsible section ([`8740b2c`](https://github.com/datajoint/element-array-ephys/commit/8740b2ca2039a1a4d4fd7ab9a2effcabfbc9d7d6))

* Add citation section ([`c8ac8e6`](https://github.com/datajoint/element-array-ephys/commit/c8ac8e656ddd374d3855fc22241c304624d925dd))

* Add links to elements.datajoint.org ([`0bc69c2`](https://github.com/datajoint/element-array-ephys/commit/0bc69c217cc679a917de97587f5d2bd205e7a41a))

* Add link to elements.datajoint.org ([`d11af2f`](https://github.com/datajoint/element-array-ephys/commit/d11af2f30659b983b936b922264e6d583ce4bc86))

* Move background file to datajoint-elements repo ([`68c94c1`](https://github.com/datajoint/element-array-ephys/commit/68c94c19884dcc41f998852537e97d8b231a7ccf))

* Create copy of `ephys_acute.py` ([`b66109b`](https://github.com/datajoint/element-array-ephys/commit/b66109b5e61297a10c1cc8a929115fa5955238e1))

* Move background file to datajoint-elements repo ([`f8a3abf`](https://github.com/datajoint/element-array-ephys/commit/f8a3abfa0af6abffec43654b0c75ae32c45c71c3))

* Merge pull request #58 from kabilar/main

Add attributes and rename module ([`1f7a2a3`](https://github.com/datajoint/element-array-ephys/commit/1f7a2a36fe0162c0ba6241f18ff34338b323e854))

* Ensure backwards compatibility ([`fc38bb5`](https://github.com/datajoint/element-array-ephys/commit/fc38bb588f03fe6971bd269a19498b50b7c2d6c7))

* Update string formatting ([`0d56f2e`](https://github.com/datajoint/element-array-ephys/commit/0d56f2e62be96fd0a94b6a43142c0cb620b16fb7))

* Rename module `ephys` to `ephys_acute` ([`1104ab4`](https://github.com/datajoint/element-array-ephys/commit/1104ab4b8820c883857e7412374c20bbc2b27689))

* Add recording metadata ([`65b9ece`](https://github.com/datajoint/element-array-ephys/commit/65b9ece60e0ec988ed201b9d4d036e1c1535fb7a))

* Merge branch &#39;main&#39; of https://github.com/datajoint/element-array-ephys into main ([`b13995e`](https://github.com/datajoint/element-array-ephys/commit/b13995e230732f8e6a3b957b1388d0d82e79a274))

* Merge pull request #35 from kabilar/main

Implement `find_full_path` within `ephys` modules ([`088093d`](https://github.com/datajoint/element-array-ephys/commit/088093d068c444311940a195f4b7408bbf0db429))

* Increase datatype size ([`6b17239`](https://github.com/datajoint/element-array-ephys/commit/6b1723940edf595329235c4805bfcc5b1b6544a8))

* Rename package ([`6f9507c`](https://github.com/datajoint/element-array-ephys/commit/6f9507c3d752f5bd010e1cf24a7369060d77b8d7))

* Merge branch &#39;main&#39; of https://github.com/kabilar/element-array-ephys into main ([`ce6adf1`](https://github.com/datajoint/element-array-ephys/commit/ce6adf1c6a7409ecd98907f3a3a5c6d50f0c10bb))

* Suggested adds re upstream components

Directing to workflow for upstream `SkullReference` and utility functions ([`4ca9b32`](https://github.com/datajoint/element-array-ephys/commit/4ca9b328f705d9363dd70a88ad857c5994f65d77))

* Update error message ([`09e8a96`](https://github.com/datajoint/element-array-ephys/commit/09e8a96504898f7840b031d09ce9346b639b600f))

* Remove print statement ([`1a4a7f5`](https://github.com/datajoint/element-array-ephys/commit/1a4a7f5c0834f62a64a6508de0d0b5de148a4657))

* [WIP] Add print statement ([`84bb616`](https://github.com/datajoint/element-array-ephys/commit/84bb6169c98fedfea50418a10c31e870b1e8913f))

* Fix for missing `fileTimeSecs` ([`665cc28`](https://github.com/datajoint/element-array-ephys/commit/665cc287b5b84cfe961bca3e47c9ff407483a2b9))

* Update module import ([`818cc53`](https://github.com/datajoint/element-array-ephys/commit/818cc53edb5395a1cc845958a373365679174f22))

* Fixed doc string ([`9881350`](https://github.com/datajoint/element-array-ephys/commit/98813508f9a77ee3110d8df055957308361273d5))

* Update module import ([`139e99b`](https://github.com/datajoint/element-array-ephys/commit/139e99b4d8dfec9c267dd8718b533cdb5a59bc00))

* Fix module import ([`44be355`](https://github.com/datajoint/element-array-ephys/commit/44be35568edfab666d48dcaa30a02e72ea65159f))

* Remove test print statement ([`cf533a2`](https://github.com/datajoint/element-array-ephys/commit/cf533a275ca1220136472bbdf30048dc9f8c92e9))

* [WIP] Add print statement ([`b98192b`](https://github.com/datajoint/element-array-ephys/commit/b98192b8ca2cde9e2babbc48b383673a5ae15a94))

* [WIP] Update directory path ([`49c554b`](https://github.com/datajoint/element-array-ephys/commit/49c554bea6a2431140b553098f889f717600da3a))

* Update comments ([`ab426c1`](https://github.com/datajoint/element-array-ephys/commit/ab426c1ed9ea14b960bcd3d3e1970c74ef020143))

* Fix OpenEphys session path ([`2233c5d`](https://github.com/datajoint/element-array-ephys/commit/2233c5ddff6351b541125b53cb6c49a424cacc72))

* [WIP] Print directory path ([`68ef14b`](https://github.com/datajoint/element-array-ephys/commit/68ef14b180c7fd5d61bbac1a9d6ec9d4a7c0530e))

* Fix import ([`2be1f08`](https://github.com/datajoint/element-array-ephys/commit/2be1f08af1d428570f5155f7d11463646805886b))

* Update author ([`b6b39c0`](https://github.com/datajoint/element-array-ephys/commit/b6b39c093a7603eba1a40b9b3b82db1c6294aac9))

* Add element_data_loader for multiple root dirs ([`ffaf60b`](https://github.com/datajoint/element-array-ephys/commit/ffaf60b72b648229b47e76ff9bb75ddedd56ef13))

* Move functions to `element-data-loader` ([`4f4be8d`](https://github.com/datajoint/element-array-ephys/commit/4f4be8d264398c3251baae5edc9be37a97c3f753))

* Merge branch &#39;main&#39; of https://github.com/datajoint/element-array-ephys into main ([`a0f49d2`](https://github.com/datajoint/element-array-ephys/commit/a0f49d27cacaceaf54a688abb4e9de2a153574a8))

* Merge pull request #30 from ttngu207/chronic_and_acute

chronic and acute probe insertions as different python modules ([`1fdbcf1`](https://github.com/datajoint/element-array-ephys/commit/1fdbcf12d1a518e686b6b79e9fbe77b736cb606a))

* rename to `ephys_chronic` ([`7474f8f`](https://github.com/datajoint/element-array-ephys/commit/7474f8f2358b784d133277ecd1da8e687ab5fa14))

* Merge branch &#39;main&#39; of https://github.com/ttngu207/element-array-ephys into chronic_and_acute ([`f28f0c6`](https://github.com/datajoint/element-array-ephys/commit/f28f0c6f489566e3c0be9cb8235ba7ea80d716f2))

* Update Dockerfile ([`d126bc5`](https://github.com/datajoint/element-array-ephys/commit/d126bc53476c5687dfcaf63689d74c18b707cf3e))

* chronic and acute probe insertions as different python modules ([`c4a9ab8`](https://github.com/datajoint/element-array-ephys/commit/c4a9ab8214c23c3c61f5f41ffef1c529e2c82b59))

* Merge branch &#39;main&#39; of https://github.com/datajoint/element-array-ephys into main ([`6472c19`](https://github.com/datajoint/element-array-ephys/commit/6472c19b6b21ffe091c4098ec848802453a63c3d))

* Merge pull request #27 from ttngu207/main

beta 0 release ([`7c67f65`](https://github.com/datajoint/element-array-ephys/commit/7c67f65bcc8844eed429ab5b5a10f457162f9f15))

* beta 0 release ([`69a5424`](https://github.com/datajoint/element-array-ephys/commit/69a5424f0404eae379978a7f990b677d62aa42b9))

* Merge pull request #26 from ttngu207/main

bump alpha version for first release on PyPI ([`7cd92ee`](https://github.com/datajoint/element-array-ephys/commit/7cd92ee693779a03abe73f91fca267a82435e59a))

* bump alpha version for first release on PyPI ([`99ab2fa`](https://github.com/datajoint/element-array-ephys/commit/99ab2fa6033e14c2ea98a4fced629287042766a6))

* Merge pull request #25 from ttngu207/main

bump alpha version for first release on pypi ([`159a2a5`](https://github.com/datajoint/element-array-ephys/commit/159a2a5d4befc1748ab11f5443239a09df759ea6))

* bump alpha version for first release on pypi ([`ab3cfc9`](https://github.com/datajoint/element-array-ephys/commit/ab3cfc922bb76e0d6e3a0930ba3f995a47891802))

* Merge pull request #24 from ttngu207/main

update README, improve markdown formatting, specify `long_description_content_type` to markdown, add versioning and GH Action for PyPI release ([`07f858c`](https://github.com/datajoint/element-array-ephys/commit/07f858c36437a7a79d8a5bddb49d026c03f274ad))

* Apply suggestions from code review

Co-authored-by: Dimitri Yatsenko &lt;dimitri@vathes.com&gt; ([`98753ed`](https://github.com/datajoint/element-array-ephys/commit/98753ed3c3a6cff88f445b633d47b8e27fb4f7df))

* minor code cleanup ([`d68d53e`](https://github.com/datajoint/element-array-ephys/commit/d68d53e28eebf51a6f44c53a591b01ee0a894e54))

* versioning and GH Action for PyPI release ([`0ffc885`](https://github.com/datajoint/element-array-ephys/commit/0ffc88595b1365992e94f6b53fe9bb7b0d4a75c4))

* update diagram ([`9ece8cd`](https://github.com/datajoint/element-array-ephys/commit/9ece8cdc32b6fbe164874bf846f8c0eb26c2d8b7))

* update README, improve markdown formatting, specify `long_description_content_type` to markdown ([`3c9662b`](https://github.com/datajoint/element-array-ephys/commit/3c9662bd166d052c43c6ed5fd080bfd1c4b764ec))

* Merge branch &#39;main&#39; of https://github.com/datajoint/element-array-ephys into main ([`1ce53f3`](https://github.com/datajoint/element-array-ephys/commit/1ce53f37081b14744e625b8896df963af24cea2a))

* Merge pull request #23 from ttngu207/main

added comments to tables ([`f2ac602`](https://github.com/datajoint/element-array-ephys/commit/f2ac602f71d9e105c584b91aa3b04a9cda6f931e))

* added comments to tables ([`f05e1fe`](https://github.com/datajoint/element-array-ephys/commit/f05e1fe5b1b7c992f35df9070049626a48cbcddc))

* Merge pull request #22 from ttngu207/main

bump version - 0.1.0a3 ([`6fcc31a`](https://github.com/datajoint/element-array-ephys/commit/6fcc31ac94afdf1602c9ce5190f682cded37a19b))

* Update CHANGELOG.md

Co-authored-by: Raphael Guzman &lt;38401847+guzman-raphael@users.noreply.github.com&gt; ([`8d8683a`](https://github.com/datajoint/element-array-ephys/commit/8d8683aa6c03b7c834e368820dc13222703d177f))

* bump version - 0.1.0a3 ([`f500492`](https://github.com/datajoint/element-array-ephys/commit/f50049292446b8479f4c0d3df83cee03892c85cb))

* Merge pull request #21 from ttngu207/main

GitHub Action for release process ([`283fad0`](https://github.com/datajoint/element-array-ephys/commit/283fad06c73f7b98e4f7f0d469005ded1149ad99))

* minor cleanup ([`6120883`](https://github.com/datajoint/element-array-ephys/commit/6120883e50b73ccabd7589a42971c47050c1b002))

* Apply suggestions from code review

Co-authored-by: Raphael Guzman &lt;38401847+guzman-raphael@users.noreply.github.com&gt; ([`ef3578e`](https://github.com/datajoint/element-array-ephys/commit/ef3578e3a6c1eaf3e082c0558c368b49cceb6a24))

* re-work `pkg_name` and use README as `long_description` ([`1cbc62a`](https://github.com/datajoint/element-array-ephys/commit/1cbc62aaf9ba42534666e4790debdd3eba5a88d4))

* add docker-compose to gitignore ([`fc8f72b`](https://github.com/datajoint/element-array-ephys/commit/fc8f72b3739adbe95a5dcd822bacaef1327aa95c))

* Merge branch &#39;main&#39; of https://github.com/ttngu207/element-array-ephys into main ([`5e32e91`](https://github.com/datajoint/element-array-ephys/commit/5e32e91b5af2ff64cb119ed730508c0ac67a2f51))

* Apply suggestions from code review

Co-authored-by: Raphael Guzman &lt;38401847+guzman-raphael@users.noreply.github.com&gt; ([`7cf70d1`](https://github.com/datajoint/element-array-ephys/commit/7cf70d10e7b064cd537121e8c23480939cfeed95))

*  for testing - update twine upload to testpypi ([`ecc0ab2`](https://github.com/datajoint/element-array-ephys/commit/ecc0ab2aefbea413361371059d9fd22d190b2306))

* address review comments, add test-changelog ([`ef7b6c9`](https://github.com/datajoint/element-array-ephys/commit/ef7b6c91c417c5b2cccde7bbc4a08b8f0c5ec02e))

* Apply suggestions from code review

Co-authored-by: Raphael Guzman &lt;38401847+guzman-raphael@users.noreply.github.com&gt; ([`17dc100`](https://github.com/datajoint/element-array-ephys/commit/17dc100159947670b024f83c5e28e35567d444b3))

* Update CHANGELOG.md ([`e04f739`](https://github.com/datajoint/element-array-ephys/commit/e04f739df575ddab534e9e3b8aa26c3b2ba41cc1))

* version 0.1.0a3 ([`f433189`](https://github.com/datajoint/element-array-ephys/commit/f4331894dc804a62f660303038a831fb273a86e7))

* update setup, point back to `datajoint` github ([`c7a1940`](https://github.com/datajoint/element-array-ephys/commit/c7a194023a77cb89686bd2e3685494180eca6099))

* GH Action bugfix - bump version ([`f2c9726`](https://github.com/datajoint/element-array-ephys/commit/f2c972601a8b826d505849d74d9f7b6b7d13dcc8))

* bugfix, add SDIST_PKG_NAME ([`e8632a3`](https://github.com/datajoint/element-array-ephys/commit/e8632a3b267f8e40717c0ee457e10e35403e5777))

* improve package_name parsing ([`be26e4b`](https://github.com/datajoint/element-array-ephys/commit/be26e4b24718830b66a3fc2774b22cf1e448f2b3))

* Update development.yaml ([`ff5f5f9`](https://github.com/datajoint/element-array-ephys/commit/ff5f5f900a6beddd835c7d6af366b563cd8f31f8))

* Update development.yaml ([`f847aeb`](https://github.com/datajoint/element-array-ephys/commit/f847aebcdb1b099dd987f292d34891c065d71ffb))

* add `build` to GH action ([`5052b8e`](https://github.com/datajoint/element-array-ephys/commit/5052b8e7e494f2397bbc0a108ef5e9825c37206f))

* change package url - for testing GH release only ([`77f5240`](https://github.com/datajoint/element-array-ephys/commit/77f524093c69ce371627f12a386efa78332b779f))

* update changelog - bump version to 0.1.0a3 ([`bae08ad`](https://github.com/datajoint/element-array-ephys/commit/bae08ad5b7995b1246f0ead8f456ebe164f68053))

* Update development.yaml ([`d124407`](https://github.com/datajoint/element-array-ephys/commit/d1244077f55ac65ddddd5335a7684eb13280e37c))

* Update .gitignore ([`b8eb640`](https://github.com/datajoint/element-array-ephys/commit/b8eb64025e65bd49081d9d7c9e93a7ad3e8dc7fd))

* set up release processes for GH Action ([`a94e726`](https://github.com/datajoint/element-array-ephys/commit/a94e7268ca99d33dee0c15d1aac1f47747ca7bfd))

* Merge branch &#39;main&#39; into GH-action-PyPI-release ([`b991fbb`](https://github.com/datajoint/element-array-ephys/commit/b991fbb776b71a5feff09b638f171488b78bd3ee))

* add `package_version` ([`02f5387`](https://github.com/datajoint/element-array-ephys/commit/02f5387a7bd4bce569193fecf09198456ea3b7b1))

* Create CHANGELOG.md ([`77a7a52`](https://github.com/datajoint/element-array-ephys/commit/77a7a5293f3d97a091a3ba159a87fe08967f9125))

* Merge pull request #20 from ttngu207/main

table renames, code cleanup ([`4cced0e`](https://github.com/datajoint/element-array-ephys/commit/4cced0edd25ef2186dd9498c67632363cf37eab2))

* table renames, code cleanup ([`236301a`](https://github.com/datajoint/element-array-ephys/commit/236301ab821107e37d26d59ee00e643c10e7f8d6))

* Merge pull request #19 from ttngu207/main

Code cleanup/optimization, variables renaming for clarity ([`b0fa79f`](https://github.com/datajoint/element-array-ephys/commit/b0fa79ff1895e963f9f4ba6b11160ca4df2d087b))

* minor bugfix ([`9b18415`](https://github.com/datajoint/element-array-ephys/commit/9b184159115cd48a60c32f6d406675caeed8147a))

* split `find_valid_full_path` to `find_full_path` and `find_root_directory` ([`258839b`](https://github.com/datajoint/element-array-ephys/commit/258839b3a97c03cccbf36deeaa7637724af98bb5))

* support `.xlsx` cluster files ([`4e824cf`](https://github.com/datajoint/element-array-ephys/commit/4e824cfbef5cfee6555c77e0341ef069bd174703))

* minor wording fix ([`855f8eb`](https://github.com/datajoint/element-array-ephys/commit/855f8eb60c8f9ac3411badaff7fa20ea7d908caa))

* remove `get_clustering_root_data_dir()` from docstring ([`6f01562`](https://github.com/datajoint/element-array-ephys/commit/6f01562f8c2b60a497be89474958956e962171c8))

* allow root_dir to be a list of potential directories - util function `find_valid_full_path()` for root and path searching ([`6488fee`](https://github.com/datajoint/element-array-ephys/commit/6488fee6499a995756a303416740809b1b5886a7))

* code refactor - improve logic for `spikeglx_meta` file search ([`e51113b`](https://github.com/datajoint/element-array-ephys/commit/e51113b94bca3267f2870936c34d1140ceb037f9))

* Update version.py ([`91a3824`](https://github.com/datajoint/element-array-ephys/commit/91a382435fc5af8021718e54d57c908a1dc30418))

* bugfix ([`669c6e5`](https://github.com/datajoint/element-array-ephys/commit/669c6e53e0882b062cc23969a855205e906c2af0))

* improve variables naming in kilosort reader ([`e761501`](https://github.com/datajoint/element-array-ephys/commit/e7615017d168e1360fe0ab7e40c0958d9a9c97e4))

* improve variables naming ([`c002646`](https://github.com/datajoint/element-array-ephys/commit/c0026467259cfff561b41fabf7ce5d08e4352911))

* improve naming, comments ([`cd28d9b`](https://github.com/datajoint/element-array-ephys/commit/cd28d9b43e319777a0e97f6e79d403623902cb06))

* code-cleanup - variables renaming - addressing PR review&#39;s comments ([`eb7eb2c`](https://github.com/datajoint/element-array-ephys/commit/eb7eb2c4336fa7e4ed8d109e24e2eba02341b8f0))

* Merge pull request #17 from ttngu207/main

specify a separate `get_clustering_root_data_dir()` - handle cases where raw ephys and clustering results are stored at different root locations (e.g. different mount points) ([`74a7a56`](https://github.com/datajoint/element-array-ephys/commit/74a7a5669f0aad4be3b430f93dd3efaad24af920))

* Merge branch &#39;main&#39; of https://github.com/ttngu207/element-array-ephys into main ([`99d761f`](https://github.com/datajoint/element-array-ephys/commit/99d761fd17b4fb410f5729a380424424f1fe5d43))

* Apply suggestions from code review - improve docstring/comments

Co-authored-by: shenshan &lt;shenshanpku@gmail.com&gt; ([`6f8cd8b`](https://github.com/datajoint/element-array-ephys/commit/6f8cd8b832af8861ac29f47ffae13036e1a90b36))

* remove Quality Control - will add this as a separate element ([`33a421b`](https://github.com/datajoint/element-array-ephys/commit/33a421b5298c784111a5e62613a1e8a018c48c1c))

* comment fix ([`af54831`](https://github.com/datajoint/element-array-ephys/commit/af54831d29596758c871d81128bc3a501cb25a98))

* naming bugfix ([`75d31a5`](https://github.com/datajoint/element-array-ephys/commit/75d31a5c36ecb575171ee77c7aecb31348533c08))

* rename &#34;OpenEphys&#34; to &#34;Open Ephys&#34; ([`bc2f528`](https://github.com/datajoint/element-array-ephys/commit/bc2f528b0bc8a94f21eb21f2c65d6fa8c5b5a409))

* added `QualityControl` as a master-table and make `ClusterQualityMetrics` the part-table now -  no need for modified `key_source` ([`2c9a787`](https://github.com/datajoint/element-array-ephys/commit/2c9a787950c7ae97f87c4a1dc998565bdb0a65ee))

* Waveform table now a master table, with Waveform.Unit as part-table (no need for modified `key_source`) ([`31e2320`](https://github.com/datajoint/element-array-ephys/commit/31e2320955525b919246bd5aae6f85ef25ec30b7))

* openephys loader - code cleanup ([`033240f`](https://github.com/datajoint/element-array-ephys/commit/033240f97a049f8b6c23d17df4548252d89ae70d))

* creating Neuropixels probe (in ProbeType) as part of `probe` module activation ([`655115b`](https://github.com/datajoint/element-array-ephys/commit/655115bcc7a4530fea488d9737c12be42da046ee))

* tweaks to LFP and waveform ingestion - do in small batches to mitigate memory issue ([`dba0a48`](https://github.com/datajoint/element-array-ephys/commit/dba0a48726553913a77b6a65b9540713da505b73))

* minor updates to &#34;jsiegle&#34; PR - code, variable-naming cleanup

Handle cases where the plugin is `Neuropix-PXI` but `NP_PROBE` is not present in `processor[&#39;EDITOR&#39;]` (only `PROBE`) ([`dcf8906`](https://github.com/datajoint/element-array-ephys/commit/dcf89069aa289d300779cb34bb4c3940be535bef))

* improve docstring/description ([`bebec1a`](https://github.com/datajoint/element-array-ephys/commit/bebec1ac563fc623fcc92125943ec410905230b2))

* &#39;enable_python_native_blobs&#39; = True ([`fcb5983`](https://github.com/datajoint/element-array-ephys/commit/fcb5983a77981182670d3294cac60d82bc9bc501))

* Merge branch &#39;main&#39; of https://github.com/datajoint/element-array-ephys into main ([`2adf2e0`](https://github.com/datajoint/element-array-ephys/commit/2adf2e06af7a89120d8d07cfb33926f216420bf2))

* Merge pull request #16 from jsiegle/main

Update Open Ephys data reader ([`cf39185`](https://github.com/datajoint/element-array-ephys/commit/cf391854d0510ec1d4a903c478f880250523780a))

* Update Open Ephys data reader ([`a85e835`](https://github.com/datajoint/element-array-ephys/commit/a85e83532f017da405ac67fd7e5d135a52d07a9f))

* specify a separate `get_clustering_root_data_dir()` - handle cases where raw ephys and clustering results are stored a different root locations (e.g. different mount points) ([`ce90dc6`](https://github.com/datajoint/element-array-ephys/commit/ce90dc6b212d64ea158c6392390f813cfa7a4df7))

* add `version.py` ([`4185ba3`](https://github.com/datajoint/element-array-ephys/commit/4185ba3adb0ccfeac6e87b1220045ca5d8753fd7))

* Update .gitignore ([`f69e491`](https://github.com/datajoint/element-array-ephys/commit/f69e491c56e4e87b9a35c922a55a19658bd82030))

* Merge pull request #13 from ttngu207/main

Mostly code cleanup - formatting ([`f07d131`](https://github.com/datajoint/element-array-ephys/commit/f07d13106edafdad87ae28f229907ff6847982e3))

* more code-cleanup ([`ea1547f`](https://github.com/datajoint/element-array-ephys/commit/ea1547fb4b31f748b9f5c2f92d622fcb752b1d73))

* Update setup.py ([`15ca803`](https://github.com/datajoint/element-array-ephys/commit/15ca803536aba0bb292d5ed3942ed85e1d4793e9))

* Update Background.md ([`0833d10`](https://github.com/datajoint/element-array-ephys/commit/0833d106cf4a6de9b0eb8acc9a5bf372fde1b979))

* cleanup ([`6c490f8`](https://github.com/datajoint/element-array-ephys/commit/6c490f8a96a09d0e681c63b36951d4def1fcbd7b))

* Update README.md ([`40ce9e6`](https://github.com/datajoint/element-array-ephys/commit/40ce9e68c5b88d842135390fa8378bb42d2a9947))

* rename `elements-ephys` -&gt; `element-array-ephys` ([`fa369f0`](https://github.com/datajoint/element-array-ephys/commit/fa369f04c43e5f6e7cb68870bf58a1d8910888e0))

* Update README.md ([`a573e5c`](https://github.com/datajoint/element-array-ephys/commit/a573e5c257623bbdf93f29ec4d9a2184feab3162))

* Update Background.md ([`cf2f172`](https://github.com/datajoint/element-array-ephys/commit/cf2f172c2a6ee5729d913e4f882c7a7d3b30168d))

* Update Background.md ([`dfff966`](https://github.com/datajoint/element-array-ephys/commit/dfff966bf190d4d9d41bd6150346b52d44edf30b))

* added Background section ([`653b84f`](https://github.com/datajoint/element-array-ephys/commit/653b84f73b8131733cb33546fd3234e85078b800))

* code cleanup - formatting ([`7ab0c2a`](https://github.com/datajoint/element-array-ephys/commit/7ab0c2a4c1cca04be1271b593d1e944a565a64b3))

* Create CONTRIBUTING.md ([`1ee37ab`](https://github.com/datajoint/element-array-ephys/commit/1ee37ab341bd0959aea572321168646e9cc97dbf))

* Merge pull request #10 from ttngu207/main

Ephys pipeline with support for multiple curations ([`983d61a`](https://github.com/datajoint/element-array-ephys/commit/983d61a89ccc42f114a51915261b443e2c2b153e))

* update diagrams ([`e98b34f`](https://github.com/datajoint/element-array-ephys/commit/e98b34f52aaa0a042810685f8b896c2288774131))

* Update requirements.txt ([`bab8e1d`](https://github.com/datajoint/element-array-ephys/commit/bab8e1d5cfbd0930323b8716d4eb80550a106bda))

* bugfix in spikeglx get original channels ([`f8244c8`](https://github.com/datajoint/element-array-ephys/commit/f8244c89ab86d83abad5ef870639180d6a751c4d))

* Merge branch &#39;multiple-curations&#39; into main ([`bfab1dd`](https://github.com/datajoint/element-array-ephys/commit/bfab1dde4dc7b3620bd4cd0950460da71ac18a2e))

* bugfix in Unit ingestion ([`adfd5af`](https://github.com/datajoint/element-array-ephys/commit/adfd5af9632f7987a427d7ff07d926e85f90bff3))

* added a `CuratedClustering` as master table for `Unit` ([`7bd751a`](https://github.com/datajoint/element-array-ephys/commit/7bd751a8bc2574b14180eb39016cdb620358c4a5))

* Update openephys.py ([`a889407`](https://github.com/datajoint/element-array-ephys/commit/a8894072c7d84e375ef9ca458d7556703916bfaf))

* minor code cleanup ([`b0011a1`](https://github.com/datajoint/element-array-ephys/commit/b0011a18ee199afb878bbf8c152d526331a2a820))

* `Curation` downstream from `Clustering` - move `Curation` insertion in `Clustering.make()` to a separate utility function ([`6859e52`](https://github.com/datajoint/element-array-ephys/commit/6859e52ba4832f7dd714c3890552b243ecffd6c7))

* Merge branch &#39;main&#39; into multiple-curations ([`64bd47d`](https://github.com/datajoint/element-array-ephys/commit/64bd47d72aa3bcd0de31d22a46db5be821ce88f1))

* prototype design for multiple curations ([`94686f5`](https://github.com/datajoint/element-array-ephys/commit/94686f5d2237f16a7cb9885f0ffb6fc11db49785))

* Merge pull request #9 from ttngu207/main

keep `_timeseries` data as memmap int16 type, apply bitvolt conversion at LFP/Waveform extraction step &amp; Bugfix in channel matching for SpikeGLX ([`70a813b`](https://github.com/datajoint/element-array-ephys/commit/70a813b207bba72bb3a268a797ef156a53c15c7a))

* Update elements_ephys/readers/spikeglx.py

Co-authored-by: Dimitri Yatsenko &lt;dimitri@vathes.com&gt; ([`93ea01a`](https://github.com/datajoint/element-array-ephys/commit/93ea01a9bad217fad18a77a99b2df46b0986828c))

* minor formatting, PEP8 ([`d656108`](https://github.com/datajoint/element-array-ephys/commit/d65610889bba20fe468c5a97663769c3a97cf418))

* datajoint version 0.13+ required ([`39580e1`](https://github.com/datajoint/element-array-ephys/commit/39580e14f2ffc0d3772c3267e5525e8f9216a5b4))

* bugfix - openephys waveform extraction ([`825407c`](https://github.com/datajoint/element-array-ephys/commit/825407c5f3fae0def1291dfaf6b87bdaf14ea5f4))

* bugfix ([`4afc0f1`](https://github.com/datajoint/element-array-ephys/commit/4afc0f11e164281b91357f3ac07b8fb3d17cbce8))

* try-catch for searching/loading spikeglx files ([`f3d98b3`](https://github.com/datajoint/element-array-ephys/commit/f3d98b3b14a903c962037ca5406c9a3302475de3))

* keep `_timeseries` data as memmap int16 type, apply bitvolt conversion only when needed (at LFP or waveform extraction) ([`f9e5fc2`](https://github.com/datajoint/element-array-ephys/commit/f9e5fc291c170fcae905c9432d5f50f439a5e891))

* Update requirements.txt ([`625c630`](https://github.com/datajoint/element-array-ephys/commit/625c6307d9f9dbb97b953131a166782b230b0f4c))

* Update attached_ephys_element.svg ([`1411687`](https://github.com/datajoint/element-array-ephys/commit/1411687ad687fb75e3cb72831bbc580696d9a5ae))

* added svg diagram ([`7a0762c`](https://github.com/datajoint/element-array-ephys/commit/7a0762c18c28acdd4009c012eabd8d102b816f76))

* Merge pull request #8 from ttngu207/main

ClusteringTask as manual table - Ingestion support for OpenEphys ([`f76086c`](https://github.com/datajoint/element-array-ephys/commit/f76086c611428ed4d8cc52edee6b240fb805779a))

* bugfix: Imax per probe type ([`56f8fdc`](https://github.com/datajoint/element-array-ephys/commit/56f8fdc43db8b1975a4cca46c1702a8670a190c2))

* code cleanup - renamed `data` -&gt; `timeseries` ([`6d5ee8b`](https://github.com/datajoint/element-array-ephys/commit/6d5ee8bf68bfc45f9a760515fb399945c85fb6be))

* code cleanup, added docstring &amp; comments to code blocks ([`e64dafe`](https://github.com/datajoint/element-array-ephys/commit/e64dafedd53de6ebf2243d6049982738f0e8d56b))

* Update spikeglx.py ([`238a511`](https://github.com/datajoint/element-array-ephys/commit/238a511d0030299b650868c78de05e428739a3e0))

* bugfix in waveform extraction ([`60e320d`](https://github.com/datajoint/element-array-ephys/commit/60e320d7973490bf3ae77ec0b6c9b86addbab921))

* added comment ([`be82f4e`](https://github.com/datajoint/element-array-ephys/commit/be82f4e9a5a262f73c427b5996bf7b3778e105ba))

* minor code cleanup ([`8aa11e2`](https://github.com/datajoint/element-array-ephys/commit/8aa11e231140f81fc10347f08a9f46e8c1e345b3))

* extract and apply bit-volts conversion for spikeglx loader ([`b5c11f0`](https://github.com/datajoint/element-array-ephys/commit/b5c11f04ae9b7b33fe93efd24fb292090e683d89))

* apply channels&#39; gain for the data ([`8ceeb0b`](https://github.com/datajoint/element-array-ephys/commit/8ceeb0b8daea0f4d6d3c1aadf28930b50ae9fec9))

* remove `used_in_reference` in ElectrodeConfig

this is misleading as it&#39;s only relevant for SpikeGLX acquisition for denoting channel visualization ([`847eeba`](https://github.com/datajoint/element-array-ephys/commit/847eeba4263c5a050ca7ffafa0cd4e891e099b21))

* bugfix in waveform extraction for OpenEphys ([`281e37b`](https://github.com/datajoint/element-array-ephys/commit/281e37b8c4c2da28fb7c94525be0db1b8eb495d4))

* bugfix in waveform ingestion ([`3452ab7`](https://github.com/datajoint/element-array-ephys/commit/3452ab721f0dc2022d1aaae0cb919e97cc25a8f8))

* code cleanup ([`3784238`](https://github.com/datajoint/element-array-ephys/commit/3784238c6ae8ed3b2c556544c054c3ca15e59e86))

* waveform ingestion for OpenEphys ([`1d02cf5`](https://github.com/datajoint/element-array-ephys/commit/1d02cf57ea04ced7a5b8062873069d5b4c473c72))

* extract_spike_waveforms() for OpenEphys ([`2d6f22c`](https://github.com/datajoint/element-array-ephys/commit/2d6f22c0a78a0ef7c617322fcc4658c045341ee1))

* implement &#34;probe&#34; in OpenEphys as a standalone class ([`045344d`](https://github.com/datajoint/element-array-ephys/commit/045344dc6ac38bf6482208065f95ff0b28aeedb9))

* minor bugfix in channel mapping/fetching ([`631837d`](https://github.com/datajoint/element-array-ephys/commit/631837d4e4f1c52a45105ee1817f397221a304cd))

* Update spikeglx.py ([`af2831c`](https://github.com/datajoint/element-array-ephys/commit/af2831ce1e4e278315644a2a7e5aab29fa495131))

* minor naming bugfix ([`e9d60d7`](https://github.com/datajoint/element-array-ephys/commit/e9d60d7088cc54b29c2b13ec5c5886fd77e5004a))

* rename `neuropixels` -&gt; `spikeglx` ([`07982dc`](https://github.com/datajoint/element-array-ephys/commit/07982dc934a1103bdca1369da621cec393b26eea))

* LFP ingestion for OpenEphys ([`75149b3`](https://github.com/datajoint/element-array-ephys/commit/75149b3a9a6f3eed51977398ef037fbfe5de27ca))

* EphysRecording&#39;s `make()` handles OpenEphys ([`f784f12`](https://github.com/datajoint/element-array-ephys/commit/f784f12373eed355f5c45d16796ba9363abc75be))

* Update probe.py ([`628c7f0`](https://github.com/datajoint/element-array-ephys/commit/628c7f06bf4764f25c0f9113474e4cb1739e3f01))

* update ephys ingestion routine, refactor electrode config generation ([`2750aa9`](https://github.com/datajoint/element-array-ephys/commit/2750aa98b861d6426d7ee9335db7c81412f4ace0))

* openephys loader, using pyopenephys pkg ([`5540bbe`](https://github.com/datajoint/element-array-ephys/commit/5540bbe9a09fc5bc287f5973eff00f3766b9e8c3))

* Update neuropixels.py ([`eba6b8c`](https://github.com/datajoint/element-array-ephys/commit/eba6b8c1303fee1541b19f3ad72a4a88e54a18b3))

* openephys loader, using `open_ephys` pkg ([`a2ba6d6`](https://github.com/datajoint/element-array-ephys/commit/a2ba6d63753e9e302427728df6c23d74a45370a6))

* Update LICENSE ([`e29180f`](https://github.com/datajoint/element-array-ephys/commit/e29180fac4da203b47540c8f358bc489ba341993))

* Update openephys.py ([`2545772`](https://github.com/datajoint/element-array-ephys/commit/25457726e2faa3a8748ec7410e0e7a6b708b8cbc))

* `ClusteringTask` as manual table with user specified paramset_idx and clustering_output_dir ([`6850702`](https://github.com/datajoint/element-array-ephys/commit/6850702d2be133942391597c77805555fcca4216))

* Merge branch &#39;main&#39; into OpenEphys-support ([`7d827a1`](https://github.com/datajoint/element-array-ephys/commit/7d827a11b8cea2952bb0a4d44b6285f1bd052ad9))

* infer/store ephys-recording directory, based on `session_dir` ([`38927c2`](https://github.com/datajoint/element-array-ephys/commit/38927c242c0a92323fb4080dac463b7e3ab3c693))

* Merge branch &#39;main&#39; of https://github.com/datajoint/elements-ephys into main ([`8a16bf2`](https://github.com/datajoint/element-array-ephys/commit/8a16bf21ea6b6213b38fff0af7328a44195f2040))

* added AcquisitionSoftware ([`7de2127`](https://github.com/datajoint/element-array-ephys/commit/7de2127e0601c7817cf77fd993f6402729840ca5))

* minor bugfix: `probe.schema.activate` -&gt; `probe.activate` ([`e278573`](https://github.com/datajoint/element-array-ephys/commit/e278573cd5c250ba9801ec2432f09e647ecb2428))

* Create open_ephys.py ([`a28c2da`](https://github.com/datajoint/element-array-ephys/commit/a28c2dac483c4f3366b185dfbd47b4c28c1f4e04))

* Merge pull request #7 from ttngu207/main

update docstring for function `activate` ([`8893fc8`](https://github.com/datajoint/element-array-ephys/commit/8893fc800bfb224e28013d1475c19f59c669ea8d))

* update wording, `required_module` -&gt; `linking_module` ([`071bf35`](https://github.com/datajoint/element-array-ephys/commit/071bf353e4376623298826af3187e0fc6c3837fa))

* update docstring for function `activate` ([`f11900f`](https://github.com/datajoint/element-array-ephys/commit/f11900f19d6735c0fd4bb4420a05d03670fd6b4e))

* Merge pull request #6 from ttngu207/main

implement new &#34;activation&#34; mechanism -&gt; using dict, module name or module for `requirement` ([`ec58e20`](https://github.com/datajoint/element-array-ephys/commit/ec58e20962689f2d87373209acd4bf07178bfeec))

* simplify &#34;activate&#34; no explicit requirements check ([`aa4064c`](https://github.com/datajoint/element-array-ephys/commit/aa4064cd22704fdb69b83ce6d793c1ba307b1a3a))

* minor format cleanup ([`822c5b7`](https://github.com/datajoint/element-array-ephys/commit/822c5b742e74ff36861ddd8a652b9bdd48bd03d8))

* implement new &#34;activation&#34; mechanism -&gt; using dict, module name or module as &#39;requirement&#39; ([`c9e7f1e`](https://github.com/datajoint/element-array-ephys/commit/c9e7f1e0b1cfbc77c6f7cffbb93b1bafbeeed731))

* bugfix in `paramset_name` -&gt; `paramset_idx` ([`852f5a4`](https://github.com/datajoint/element-array-ephys/commit/852f5a471f0ed0703c774716c46f5484854e9e57))

* Merge pull request #5 from ttngu207/main

minor tweak using `schema.database`, awaiting `schema.is_activated` ([`e9191dd`](https://github.com/datajoint/element-array-ephys/commit/e9191dd6c9c225874aa046ea75c0ea0acc581c17))

* minor tweak using `schema.database`, awaiting `schema.is_activated` ([`d606233`](https://github.com/datajoint/element-array-ephys/commit/d606233e80ab3c289c23d852977147823f8e09dc))

* Merge pull request #4 from dimitri-yatsenko/main

ephys.activate inserts required functions into the module namespace ([`dadda0d`](https://github.com/datajoint/element-array-ephys/commit/dadda0d19ecc0afb7043e8ad888d918dacca0378))

* ephys.activate inserts required functions into the module namespace ([`1f732d3`](https://github.com/datajoint/element-array-ephys/commit/1f732d39ea11d03f88147fef645f29abc59eede5))

* Merge branch &#39;main&#39; of https://github.com/datajoint/elements-ephys into main ([`3db8461`](https://github.com/datajoint/element-array-ephys/commit/3db84614d914863c9ec76752009bf8450f9439e5))

* Merge pull request #3 from ttngu207/main

code cleanup, bug fix, tested ([`1cc5119`](https://github.com/datajoint/element-array-ephys/commit/1cc51196433e8ff6046f54cbd99162d0a7ed857b))

* code cleanup, bug fix, tested ([`11d2aec`](https://github.com/datajoint/element-array-ephys/commit/11d2aec8bbaf9bf2c396d86a7f6e513607476dbf))

* Merge pull request #2 from dimitri-yatsenko/main

Refactor to use schema.activate from DataJoint 0.13 ([`5834b4a`](https://github.com/datajoint/element-array-ephys/commit/5834b4a75e38ffcd6dd1b8333f88300e0f2124cc))

* fix imported class names in the ephys module ([`b822e63`](https://github.com/datajoint/element-array-ephys/commit/b822e6313d0d73a2a4d7d26706d1ae712ad806ae))

* minor cleanup ([`5edc3ce`](https://github.com/datajoint/element-array-ephys/commit/5edc3ced1129c8807a422876b3da261c2f1d6c11))

* update to comply with datajoint 0.13 deferred schema use ([`a925450`](https://github.com/datajoint/element-array-ephys/commit/a925450db74e102dfd6a66dc55e53629a0d41765))

* Merge pull request #1 from ttngu207/main

moved from `canonical-ephys` ([`d1decf2`](https://github.com/datajoint/element-array-ephys/commit/d1decf2ac4c5e021c6c63a554052ed72ee9a1379))

* moved from `canonical-ephys` ([`55f7717`](https://github.com/datajoint/element-array-ephys/commit/55f771729d06cd9a8346d4ed0882bd51ae603489))

* Create README.md ([`0896c85`](https://github.com/datajoint/element-array-ephys/commit/0896c85193a93550e19775c7c4b02b1fa5f7742f))
