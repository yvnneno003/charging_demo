#  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#   Copyright (C) 2023 MSI-FUNTORO
#
#   Licensed under the MSI-FUNTORO License, Version 1.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       https://www.funtoro.com/global/
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import pkg_resources
from colorama import init, Fore, Style


def show_recommend_version(recommend_package_dict={}):
    init(autoreset=True)
    print('')
    print('[ Recommend Version ]')
    print('\t%-30s' % 'package' + '|' + '%-25s' % '  recommend version' + '|' + '%25s' % 'current version')
    print('\t------------------------------|-------------------------|-------------------------')
    for package_name in recommend_package_dict.keys():
        recommend_version = recommend_package_dict[package_name][2:]
        current_version = get_current_version(package_name)
        compare_version_result = compare_version(current_version, recommend_version)

        # uninstall
        if compare_version_result is None:
            print('\t%-30s' % package_name + '|' + '%-25s' % ('  ' + recommend_package_dict[package_name]) + '|' + Fore.RED + '%25s' % 'uninstall')

        else:
            if recommend_package_dict[package_name][0:2] == '==':
                if compare_version_result == 0:
                    print('\t%-30s' % package_name + '|' + '%-25s' % ('  ' + recommend_package_dict[package_name]) + '|' + Fore.GREEN + '%25s' % current_version)
                else:
                    print('\t%-30s' % package_name + '|' + '%-25s' % ('  ' + recommend_package_dict[package_name]) + '|' + Fore.RED + '%25s' % current_version)
            else:
                if compare_version_result == 1 or compare_version_result == 0:
                    print('\t%-30s' % package_name + '|' + '%-25s' % ('  ' + recommend_package_dict[package_name]) + '|' + Fore.GREEN + '%25s' % current_version)
                else:
                    print('\t%-30s' % package_name + '|' + '%-25s' % ('  ' + recommend_package_dict[package_name]) + '|' + Fore.RED + '%25s' % current_version)

    print('\n')


def get_current_version(package_name):
    try:
        current_version = pkg_resources.get_distribution(package_name).version

    except Exception as e:
        current_version = 'uninstall'

    finally:
        return current_version


def compare_version(version1, version2):
    try:
        version1 = version1.split('+')[0]
        version2 = version2.split('+')[0]

        versions1 = [int(v) for v in version1.split(".")]
        versions2 = [int(v) for v in version2.split(".")]
        for i in range(max(len(versions1), len(versions2))):
            v1 = versions1[i] if i < len(versions1) else 0
            v2 = versions2[i] if i < len(versions2) else 0
            if v1 > v2:
                return 1
            elif v1 < v2:
                return -1
        return 0
    except Exception as _:
        return None