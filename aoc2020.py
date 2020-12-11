import re
import math
from collections import defaultdict
from collections import Counter


def day1():
    print('Day 1')
    array = []
    with open('2020\input1.txt') as file:
        for data in file:
            array.append(int(data.strip('\n')))

    # part 1
    for i in array:
        temp = 2020 - i
        if temp in array:
            print(i, temp)
            print(f'Part 1: {i * temp}')
            break

    # part 2
    index = 0
    done = False
    for a in array:
        for b in array[index + 1:]:
            for c in array[index + 2:]:
                if a + b + c == 2020:
                    print(a, b, c)
                    print(f'Part 2: {a * b * c}')
                    done = True
                    break
            if done:
                break
        if done:
            break
        index += 1

    print()


def day2():
    print('Day 2')
    part_1 = 0
    part_2 = 0
    with open('2020\input2.txt') as file:
        for data in file:
            counter = 0
            data = data.split(' ')
            allow_range = (int(data[0].split('-')[0]), int(data[0].split('-')[1]))
            letter = data[1][0]
            string = data[2].split('\n')[0]

            # part 1
            for i in string:
                if i == letter:
                    counter += 1

            if allow_range[0] <= counter <= allow_range[1]:
                part_1 += 1

            # part 2
            position_1 = allow_range[0] - 1
            position_2 = allow_range[1] - 1
            if string[position_1] == letter and string[position_2] != letter:
                part_2 += 1
            elif string[position_1] != letter and string[position_2] == letter:
                part_2 += 1

        print(f'Part 1: {part_1}')
        print(f'Part 2: {part_2}')
        print()


def day3Helper(array, right, down_value):
    tree_count = 0
    start = 0
    down = 0
    while True:
        try:
            if array[down][start] == '#':
                tree_count += 1
        except IndexError:
            break
        start += right
        down += down_value
    return tree_count


def day3():
    print('Day 3')
    # part 1
    array = []
    with open('2020\input3.txt') as file:
        for data in file:
            # the map will need to expand as many times as possible to get the number that will not change the answer
            # anymore
            data = data.rstrip() * 100
            array.append(data)
    part_1 = day3Helper(array, 3, 1)
    print(f'Part 1: {part_1}')

    # part 2
    first = day3Helper(array, 1, 1)
    third = day3Helper(array, 5, 1)
    forth = day3Helper(array, 7, 1)
    fifth = day3Helper(array, 1, 2)

    part_2 = part_1 * first * third * forth * fifth

    print(f'Part 2: {part_2}')


def day4Helper(valid_data):
    valid = True
    for i in valid_data:
        key = i[:3]
        if key == 'byr':
            value = i[4:]
            if len(value) != 4 or int(value) < 1920 or int(value) > 2002:
                valid = False
                break
        elif key == 'iyr':
            value = i[4:]
            if len(value) != 4 or int(value) < 2010 or int(value) > 2020:
                valid = False
                break
        elif key == 'eyr':
            value = i[4:]
            if len(value) != 4 or int(value) < 2020 or int(value) > 2030:
                valid = False
                break
        elif key == 'hgt':
            measure = i[-2:]
            digits = [a for a in i if a.isdigit()]
            digits = int(''.join(digits))
            if measure == 'cm':
                if digits < 150 or digits > 193:
                    valid = False
                    break
            elif measure == 'in':
                if digits < 59 or digits > 76:
                    valid = False
                    break
            else:
                valid = False
                break
        elif key == 'hcl':
            value = i[4:]
            pound = value[0]
            if pound != '#':
                valid = False
                break
            if len(value) > 7:
                valid = False
                break

            pattern = re.compile('[#a-f0-9]+')
            if not pattern.fullmatch(value):
                valid = False
                break
        elif key == 'ecl':
            valid_ecl = ['amb', 'blu', 'brn', 'gry', 'grn', 'hzl', 'oth']
            value = i[4:]
            if value not in valid_ecl:
                valid = False
                break
        elif key == 'pid':
            value = i[4:]
            if len(value) != 9 or not value.isdigit():
                valid = False
                break
    return valid


def day4():
    print('Day 4')
    part_1 = 0
    part_2 = 0

    # part 1
    require = {'byr': 0, 'iyr': 0, 'eyr': 0, 'hgt': 0, 'hcl': 0, 'ecl': 0,
               'pid': 0, 'cid': 0}
    # part 2
    valid_data = []
    with open('2020\input4.txt') as file:
        for data in file:
            data = data.rstrip()
            if data != '':
                data = data.split(' ')
                for i in data:
                    key = i[:3]
                    require[key] += 1
                    valid_data.append(i)
            else:
                check = sum(require.values()) - require['cid']
                if check == 7:
                    part_1 += 1
                    # part 2
                    valid = day4Helper(valid_data)
                    if valid:
                        part_2 += 1
                valid_data = []
                require = {'byr': 0, 'iyr': 0, 'eyr': 0, 'hgt': 0, 'hcl': 0, 'ecl': 0,
                           'pid': 0, 'cid': 0}

    # check the last data because there is no blank line at the end of the text
    check = sum(require.values()) - require['cid']
    if check == 7:
        part_1 += 1
        valid = day4Helper(valid_data)
        if valid:
            part_2 += 1

    print(f'Part 1: {part_1}')
    print(f'Part 2: {part_2}')


def day5Helper(data):
    row = data[:7]
    column = data[7:]

    lower_1 = 0
    upper_1 = 127

    for i in row:
        if i == 'F':
            upper_1 = math.floor((upper_1 + lower_1) / 2)
        else:
            lower_1 = math.ceil((upper_1 + lower_1) / 2)
    row = upper_1

    lower_2 = 0
    upper_2 = 7
    for i in column:
        if i == 'L':
            upper_2 = math.floor((upper_2 + lower_2) / 2)
        else:
            lower_2 = math.ceil((upper_2 + lower_2) / 2)
    column = upper_2

    return row * 8 + column


def day5():
    print('Day 5')
    # part 1
    array = []
    with open('2020\input5.txt') as file:
        for data in file:
            array.append(day5Helper(data))

    part_1 = max(array)
    print(f'Part 1: {part_1}')

    # part 2
    part_2 = 0
    array = sorted(array)
    index = 1
    limit = len(array) - 2
    while index < limit:
        seat_id = array[index]
        if seat_id - 1 not in array:
            part_2 = seat_id - 1
            break
        elif seat_id + 1 not in array:
            part_2 = seat_id + 1
            break
        index += 1

    print(array)
    print(f'Part 2: {part_2}')


def day6():
    print('Day 6')

    part_1 = 0
    part_2 = 0

    # part 1
    with open('2020/input6.txt') as file:
        answer = set()
        for data in file:
            data = data.rstrip()
            if data != '':
                for i in data:
                    answer.add(i)
            else:
                part_1 += len(answer)
                answer = set()

    # the last line in the input file
    part_1 += len(answer)
    print(f'Part 1: {part_1}')

    # part 2
    with open('2020/input6.txt') as file:
        answer = dict()
        count = 0
        for data in file:
            data = data.rstrip()
            if data != '':
                for i in data:
                    if i not in answer:
                        answer[i] = 1
                    else:
                        answer[i] += 1
                count += 1
            else:
                for i in answer:
                    if answer[i] == count:
                        part_2 += 1
                answer = dict()
                count = 0

    # the last line in the input file
    for i in answer:
        if answer[i] == count:
            part_2 += 1

    print(f'Part 2: {part_2}')


def day7HelperPart1(bag_dict, all_bags, bag_name='shiny gold'):
    bag_to_find = []
    for i in bag_dict:
        bags = bag_dict[i]
        for bag in bags:
            name, count = bag
            if name == bag_name:
                bag_to_find.append(i)

    for i in bag_to_find:
        all_bags.add(i)
        day7HelperPart1(bag_dict, all_bags, i)


def day7HelperPart2(bag_dict, all_bags, bag_name='shiny gold'):
    to_find_bag = bag_dict[bag_name]
    for i in to_find_bag:
        if i[0] != 'no other':
            all_bags.append(i[1])
            for t in range(i[1]):
                day7HelperPart2(bag_dict, all_bags, i[0])


def day7():
    print('Day 7')

    # part 1
    bag_dict = defaultdict(list)
    bag_pattern = re.compile('bag')

    with open('2020\input7.txt') as file:
        # with open('2020\\test.txt') as file:
        for data in file:
            data = data.rstrip().split(' ')

            key_bag = ''
            temp_bag = ''
            check_key_bag = False
            temp_value = 0
            for i in data:
                if bag_pattern.match(i):
                    if len(temp_bag):
                        bag_dict[key_bag].append((temp_bag, temp_value))
                        temp_bag = ''
                        temp_value = 0
                    continue

                if i.isdigit():
                    temp_value = int(i)
                elif i != 'contain':
                    if not check_key_bag:
                        if not len(key_bag):
                            key_bag += i + ' '
                        else:
                            key_bag += i
                    else:
                        if not len(temp_bag):
                            temp_bag += i + ' '
                        else:
                            temp_bag += i
                elif i == 'contain':
                    bag_dict[key_bag] = []
                    check_key_bag = True

    all_bags = set()
    day7HelperPart1(bag_dict, all_bags)
    print(f'Part 1: {len(all_bags)}')

    # part 2
    all_bags = list()
    day7HelperPart2(bag_dict, all_bags)
    print(f'Part 2: {sum(all_bags)}')


def day8Helper(instructions, corrupted_instruction_index):
    acc = 0
    not_solved = True

    for i in corrupted_instruction_index:
        if not_solved:
            if instructions[i][0] == 'nop':
                instructions[i][0] = 'jmp'
            else:
                instructions[i][0] = 'nop'

            index = 0
            acc = 0
            not_loop = False
            while True:
                try:
                    if instructions[index][2] == '2':
                        break
                except IndexError:
                    not_loop = True
                    break

                instructions[index][2] = str(int(instructions[index][2]) + 1)

                if instructions[index][0] == 'nop':
                    index += 1
                elif instructions[index][0] == 'acc':
                    acc += int(instructions[index][1])
                    index += 1
                elif instructions[index][0] == 'jmp':
                    index += int(instructions[index][1])

            if not_loop:
                not_solved = False
            else:
                if instructions[i][0] == 'nop':
                    instructions[i][0] = 'jmp'
                else:
                    instructions[i][0] = 'nop'
                day8ResetCounter(instructions)
        else:
            break

    return acc


def day8ResetCounter(instructions):
    for i in instructions:
        i[2] = '1'


def day8():
    print('Day 8')

    array = []
    with open('2020\input8.txt') as file:
        for data in file:
            data = data.rstrip().split(' ')
            data.append('1')
            array.append(data)

    acc = 0
    index = 0
    executed_instruction = []
    while True:
        if array[index][2] == '2':
            break

        array[index][2] = str(int(array[index][2]) + 1)

        if array[index][0] == 'nop':
            executed_instruction.append(index)
            index += 1
        elif array[index][0] == 'acc':
            acc += int(array[index][1])
            index += 1
        elif array[index][0] == 'jmp':
            executed_instruction.append(index)
            index += int(array[index][1])

    day8ResetCounter(array)

    print(f'Part 1: {acc}')
    print(f'Part 2: {day8Helper(array, executed_instruction)}')


def day9():
    print('Day 9')

    array = []
    with open('2020\input9.txt') as file:
        for data in file:
            data = data.rstrip()
            data = int(data)
            array.append(data)

    # part 1
    preamble_length = 25
    index = preamble_length
    found = False
    part1 = 0
    while index < len(array):
        boundary = array[index - preamble_length:index]

        for i in boundary:
            for t in boundary[1:]:
                if i + t == array[index]:
                    found = True
                    break
        if found:
            found = False
            index += 1
        else:
            part1 = array[index]
            print(f'Part 1: {part1}')
            break

    # part 2
    index = 0
    found = False
    part2 = []
    while index < len(array):
        part2 = [array[index]]
        for i in array[index + 1:]:
            if sum(part2) + i == part1:
                found = True
                break
            elif sum(part2) + i > part1:
                break
            part2.append(i)

        if not found:
            index += 1
        else:
            break

    part2 = sorted(part2)
    print(f'Part 2: {part2[0] + part2[-1]}')


def day10Helper(array):
    if len(array) == 1:
        return 1

    return 1


def day10():
    print('Day 10')
    # part 1
    array = []
    with open('2020\input10.txt') as file:
        for data in file:
            data = data.rstrip()
            array.append(int(data))

    array = sorted(array)
    start = 0
    different_jolt_dict = {1: 0, 2: 0, 3: 0}

    for i in array:
        different_jolt_dict[i - start] += 1
        start = i

    different_jolt_dict[3] += 1

    print(f'Part 1: {different_jolt_dict[1] * different_jolt_dict[3]}')

    # part 2 credit to https://www.reddit.com/user/kaur_virunurm/
    """
        - start from wall adapter (root node) with input count 1
        - add this count to the next 1, 2 or 3 adapters / nodes
        - add their input counts to next adapters / nodes
        - repeat this for all adapters (in sorted order)
        - you'll end up with input count for your device adapter
    """
    array.insert(0, 0)
    c = Counter({0: 1})
    for x in array:
        c[x + 1] += c[x]
        c[x + 2] += c[x]
        c[x + 3] += c[x]
    print(f'Part 2: {c[max(array) + 3]}')
        

def day11():
    print('Day 11')


if __name__ == '__main__':
    day11()
