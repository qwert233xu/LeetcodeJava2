package com.xu;


import java.util.*;
import java.util.stream.IntStream;

// 2
public class App6 {

    public static void main(String[] args) {

    }


    // 1、 查找算法
    // 方法一：递归
    private int search_res = -1;
    public int search(int[] nums, int target) {
        search_binSearch(nums, 0, nums.length-1, target);
        return search_res;
    }


    public void search_binSearch(int[] nums, int left, int right, int target){
        int mid = (left + right) / 2;

        if (left < right){
            if (nums[mid] > target){
                search_binSearch(nums, left, mid - 1, target);
            }else if (nums[mid] < target){
                search_binSearch(nums, mid + 1, nums.length-1, target);
            }else if (nums[mid] == target){
                search_res = mid;
            }
        }
    }

    // 方法二:双指针   时间复杂度 O（logn 以 2 为底）   空间复杂度 O（1）
    public int search2(int[] nums, int target) {

        int left = 0;
        int right = nums.length - 1;

        while (left <= right){
            int mid = (left + right) / 2;
            if (nums[mid] > target){
                right = mid - 1;
            }else if (nums[mid] < target){
                left = mid + 1;
            }else {
                return mid;
            }
        }

        return -1;
    }

    // 插值查找
    // 把 1/2 比例改进为自适应  时间复杂度   O（log2（log2n））       int mid = low+(value-a[low])/(a[high]-a[low])*(high-low);


    // 反转链表
    // 1、双指针法
    public ListNode_2 reverseList(ListNode_2 head) {

        if(head == null || head.next == null) return head;

        ListNode_2 pre = null;
        ListNode_2 cur = head;

        while (cur.next != null){
            ListNode_2 cur_next = cur.next;
            cur.next = pre;
            pre = cur;
            cur = cur_next;
        }
        cur.next = pre;
        return cur;
    }

    // 2、递归写法      还可以使用头插法 以及 栈解决链表反转问题
    public ListNode_2 reverseList2(ListNode_2 head) {
        return reverseList2_rec(null, head);
    }

    public ListNode_2 reverseList2_rec(ListNode_2 pre, ListNode_2 cur) {
        if (cur == null) return pre;

        // 记录 cur 的下一个节点
        ListNode_2 cur_next = cur.next;
        // 反转操作
        cur.next = pre;
        pre = cur;
        cur = cur_next;

        return reverseList2_rec(pre, cur); // 递归进行相同的操作
    }

    // 两两交换链表中的节点   使用递归解法   O（n）的时间复杂度   或者直接使用迭代
    private ListNode swapPairsHead;
    public ListNode swapPairs(ListNode head) {
        if (head == null || head.next == null){
            return head;
        }
        swapPairsHead = head.next;
        swapPairs_rec(head, head.next);
        return swapPairsHead;
    }

    public void swapPairs_rec(ListNode pre, ListNode cur){
        if (cur == null) return;

        // 保存下一个节点
        ListNode cur_next = cur.next;
        // 反转操作
        cur.next = pre;
        // 连接操作
        if (cur_next == null){
            pre.next = null;
        }else {
            if (cur_next.next == null){
                pre.next = cur_next;
            }else {
                pre.next = cur_next.next;
                // 更新节点
                pre = cur_next;
                cur = cur_next.next;
                swapPairs_rec(pre, cur);
            }
        }
    }

    // 删除链表的倒数第 n 个节点  本次使用的 回溯   快慢指针（fast指针先移动 n+1 步，slow和fast再一起走，直到快指针为null，将slow指针下一个节点删除） 或者  栈（计算个数）？
    private int countAll = 0;
    public ListNode removeNthFromEnd(ListNode head, int n) {
        if (head == null || head.next == null) return null;

        // 虚拟头节点
        ListNode virtual = new ListNode(-1);
        virtual.next = head;
        removeNthFromEnd_backTracking(virtual, head, n, 0);
        return virtual.next;
    }

    public void removeNthFromEnd_backTracking(ListNode pre, ListNode cur, int n, int c) {
        if (cur == null) return;
        countAll++; // 计算链表节点个数
        c++; // 顺序计算索引位置
        removeNthFromEnd_backTracking(cur, cur.next, n, c);
        if (countAll - c + 1 == n) { // 找到该节点将其删除
            pre.next = cur.next;
        }
    }

   // 环形链表
   public ListNode detectCycle(ListNode head) {
        if(head == null || head.next == null) return null;

        ListNode virtual = new ListNode(-1);
        virtual.next = head;

        ListNode slow = virtual;
        ListNode fast = virtual;

        while (fast.next != null && fast.next.next != null){
            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast){
                break;
            }
        }
        if (slow!=fast){
            return null;
        }

        slow = virtual;
        while (fast != null){
            slow = slow.next;
            fast = fast.next;
            if (slow == fast){
                return slow;
            }
        }
        return null;
   }


   // 哈希表
    // 有效的字母异位词  1、数组模拟哈希表   修改时间复杂度为 O（1）   2、字典模拟哈希表      数组快于字典 hashmap底层是散列表，查询时间复杂度为 O（1）
   public boolean isAnagram(String s, String t) {

       int[] sArray = new int[26];
       int[] tArray = new int[26];

       for (char c: s.toCharArray()) {
           sArray[c - 'a'] ++;
       }

       for (char c2: t.toCharArray()) {
           tArray[c2 - 'a'] ++;
       }

       for (int i = 0; i < sArray.length; i++) {
           if (sArray[i] != tArray[i]){
               return false;
           }
       }

       return true;
   }


   // 可以只使用一个数组  进行++--操作，判断是否存在不为0的元素，是则为false，否则为true


    public boolean isAnagram2(String s, String t) {

        HashMap<Character, Integer> map1 = new HashMap<>();
        HashMap<Character, Integer> map2 = new HashMap<>();

        for (char c: s.toCharArray()) {
            map1.put(c, map1.getOrDefault(c, 0)+1);
        }

        for (char c2: t.toCharArray()) {
            map2.put(c2, map2.getOrDefault(c2, 0)+1);
        }

        if (map1.size() != map2.size()){
            return false;
        }

        Set<Map.Entry<Character, Integer>> entrySet = map1.entrySet();
        for (Map.Entry<Character, Integer> characterIntegerEntry : entrySet) {
            if (!map2.containsKey(characterIntegerEntry.getKey())){
                return false;
            }else {
                if (!Objects.equals(map2.get(characterIntegerEntry.getKey()), characterIntegerEntry.getValue())){
                    return false;
                }
            }
        }

        return true;
    }

    // 两个数组的交集
    public int[] intersection(int[] nums1, int[] nums2) {
        // 与上一题思路一样
        ArrayList<Integer> res = new ArrayList<>();

        HashMap<Integer, Integer> map1 = new HashMap<>();
        HashMap<Integer, Integer> map2 = new HashMap<>();

        for (int c: nums1) {
            map1.put(c, map1.getOrDefault(c, 0)+1);
        }

        for (int c2: nums2) {
            map2.put(c2, map2.getOrDefault(c2, 0)+1);
        }


        Set<Map.Entry<Integer, Integer>> entrySet = map1.entrySet();
        for (Map.Entry<Integer, Integer> characterIntegerEntry : entrySet) {
            if (map2.containsKey(characterIntegerEntry.getKey())){
                res.add(characterIntegerEntry.getKey());
            }
        }

        int[] result = new int[res.size()];
        int i = 0;
        for (int v: res) {
            result[i++] = v;
        }

        return result;
    }

    // 快乐数 使用set 记录是否陷入无限循环   使用 递归 + set容器
    HashSet<Integer> hashSet = new HashSet<>();
    public boolean isHappy(int n) {
        if (n == 1)return true;
        if (hashSet.contains(n)){
            return false;
        }else {
            hashSet.add(n);
        }

        int sum = 0;
        while (n / 10 != 0){
            sum += (n%10)*(n%10);
            n /= 10;
        }

        sum += n*n;
        return isHappy(sum);
    }

    // 两数之和


    public int[] twoSum(int[] nums, int target) {
        int[] res = new int[2];
        HashMap<Integer, Integer> map = new HashMap<>();

        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(target - nums[i])){
                res[0] = map.get(target - nums[i]);
                res[1] = i;
                return res;
            }
            map.put(nums[i], i);
        }

        return res;
    }

    // 四数相加II
    public int fourSumCount(int[] nums1, int[] nums2, int[] nums3, int[] nums4) {
        HashMap<Integer, Integer> map = new HashMap<>();

        int res = 0;
        for (int k : nums1) {
            for (int i : nums2) {
                map.put(k + i, map.getOrDefault(k + i, 0) + 1);
            }
        }

        for (int value : nums3) {
            for (int i : nums4) {
                if (map.containsKey(-value-i)){
                    res += map.get(-value-i);
                }
            }
        }

        return res;
    }


    // 赎金信
    public boolean canConstruct(String ransomNote, String magazine) {

        int[] ransomNote_ = new int[26];
        int[] magazine_ = new int[26];

        for (int i = 0; i < ransomNote.length(); i++) {
            ransomNote_[ransomNote.charAt(i) - 'a'] ++;
        }

        for (int i = 0; i < magazine.length(); i++) {
            magazine_[magazine.charAt(i) - 'a'] ++;
        }

        for (int i = 0; i < ransomNote.length(); i++) {
            if (ransomNote_[ransomNote.charAt(i) - 'a'] > magazine_[ransomNote.charAt(i) - 'a']){
                return false;
            }
        }

        return true;
    }

    // 字符串
    // 反转字符串
    public void reverseString(char[] s) {
        // 双指针直接交换
        int left = 0;
        int right = s.length - 1;

        while (left < right){
            char temp = s[right];
            s[right] = s[left];
            s[left] = temp;
            left++;
            right--;
        }
    }


    public String reverseString2(char[] s) {
        // 双指针直接交换
        int left = 0;
        int right = s.length - 1;

        while (left < right){
            char temp = s[right];
            s[right] = s[left];
            s[left] = temp;
            left++;
            right--;
        }

        StringBuilder res = new StringBuilder();
        for (char c : s) {
            res.append(c);
        }
        return res.toString();
    }

    // 反转字符串II
    public String reverseStr(String s, int k) {
        if (s.length() < k){
            return reverseString2(s.toCharArray());
        }

        StringBuilder res = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            // 2k的倍数
            if ((i + 1) % (2*k) == 0){
                res.append(reverseString2(s.substring((i+1-2*k), i+1-2*k+k).toCharArray()));
                res.append(s, i+1-2*k+k, i+1);
            }
        }

        if (s.length() % (2*k) > 0 && s.length() % (2*k) < k){
            // 不变
            res.append(reverseString2(s.substring(s.length() - (s.length() % (2*k))).toCharArray()));
        }else if (s.length() % (2*k) >= k && s.length() % (2*k) < 2*k){
            // 反转k
            String rear = s.substring(s.length() - (s.length() % (2*k)));
            res.append(reverseString2(rear.substring(0, k).toCharArray()));
            res.append(rear.substring(k));
        }

        return res.toString();
    }


    // 可以改进  精简逻辑

    // 填补空格
    public String replaceSpace(String s) {

//        int countOfSpace = 0;
//        for (char c : s.toCharArray()) {
//            if (c == ' '){
//                countOfSpace ++;
//            }
//        }
        StringBuilder res = new StringBuilder();
        for (char c : s.toCharArray()) {
            if (c == ' '){
                res.append("%20");
            }else {
                res.append(c);
            }
        }

        return res.toString();
    }

    // 改进之不使用额外的空间

    public String reverseWords(String s) {

        String trim = s.trim();
        // 快慢指针
        StringBuilder res = new StringBuilder();

        int slow = trim.length() - 1;
        int fast = trim.length() - 1;
        for (; fast >= 0; fast--) {
            if (trim.charAt(fast) == ' '){
                res.append(trim, fast+1, slow+1);
                res.append(" ");
                while (trim.charAt(fast) == ' '){
                    fast--;
                }
                slow = fast;
            }
        }
        res.append(trim, 0, slow+1);
        return res.toString();
    }

    // 左旋转字符串    kmp经典题目
    public String reverseLeftWords(String s, int n) {
        return s.substring(n, s.length()) +
                s.substring(0, n);
    }
    // 改进之不需要额外的空间， 则三次反转操作即可


    // 找出字符串中第一个匹配项的下标  100%用时，讨巧，后序回顾一下kmp解法
    public int strStr(String haystack, String needle) {
        for (int i = 0; i < haystack.length(); i++) {
            if (haystack.substring(i).startsWith(needle)){
                return i;
            }
        }

        return -1;
    }

    // 重复的子字符串   移动匹配解法（耗时）和kmp解法（高效）  这里使用移动匹配解法
    public boolean repeatedSubstringPattern(String s) {

        String res = s + s;
        res = res.substring(1);
        res = res.substring(0, res.length() - 1);

        return res.contains(s);
    }


    // 双指针法  前面都涉及到了

    // 栈与队列

    // 有效的括号
    public boolean isValid(String s) {
        if (s.length() == 1){
            return false;
        }
        Stack<Character> stack = new Stack<>();
        boolean res = true;
        for (char c : s.toCharArray()) {
            if (c == '(' || c == '[' || c == '{'){
                stack.push(c);
            }else {
                if (c == ')'){
                    if (stack.isEmpty()){
                        System.out.println(1);
                        return false;
                    }else {
                        char v = stack.pop();
                        if (v == '('){
//                            res = true;
                            }else {
                                res = false;
                            }
                        }
                }else if (c == ']'){
                    if (stack.isEmpty()){
                        System.out.println(3);
                        return false;
                    }else {
                        char v = stack.pop();
                        if (v == '['){
//                            res = true;
                        }else {
                            res = false;
                        }
                    }
                }else if (c == '}'){
                    if (stack.isEmpty()){
                        System.out.println(5);
                        return false;
                    }else {
                        char v = stack.pop();
                        if (v == '{'){
//                            res = true;
                        }else {
                            res = false;
                        }
                    }
                }
            }
        }
        return res && stack.isEmpty();
    }


    // 删除字符串中的所有相邻重复项
    public String removeDuplicates(String s) {
        int len = s.length();
        if (len == 1) return s;
        Stack<Character> stack = new Stack<>();
        stack.push(s.charAt(0));
        for (int i = 1; i < len; i++) {
            if (!stack.isEmpty()){
                if (stack.peek() == s.charAt(i)){
                    stack.pop();
                }else {
                    stack.push(s.charAt(i));
                }
            }else {
               stack.push(s.charAt(i));
            }
        }
        StringBuilder res = new StringBuilder();
        while (!stack.isEmpty()){
            res.insert(0, stack.pop());
        }
        return res.toString();
    }

    // 逆波兰表达式求值   一个数字栈  一个符号栈
    public int evalRPN(String[] tokens) {
        Stack<Integer> numStack = new Stack<>();

        for (String token : tokens) {
            if (!Objects.equals(token, "+") && !Objects.equals(token, "-") && !Objects.equals(token, "*") && !Objects.equals(token, "/")) {
                // 为数字
                numStack.push(Integer.parseInt(token));
            } else {
                // 计算
                int v2 = numStack.pop();
                int v1 = numStack.pop();
                switch (token) {
                    case "+":
                        numStack.push(v1 + v2);
                        break;
                    case "-":
                        numStack.push(v1 - v2);
                        break;
                    case "*":
                        numStack.push(v1 * v2);
                        break;
                    default:
                        numStack.push(v1 / v2);
                        break;
                }
            }
        }
        return numStack.pop();
    }

    // 滑动窗口最大值   单调队列
    public int[] maxSlidingWindow(int[] nums, int k) {

        if (nums.length == 1) {
            return nums;
        }
        int len = nums.length - k + 1;
        //存放结果元素的数组
        int[] res = new int[len];
        int num = 0;
        //自定义队列
        MyPorQueue myQueue = new MyPorQueue();
        //先将前k的元素放入队列
        for (int i = 0; i < k; i++) {
            myQueue.add(nums[i]);
        }
        res[num++] = myQueue.peek(); // 前k个元素
        for (int i = k; i < nums.length; i++) {
            //滑动窗口移除最前面的元素，移除是判断该元素是否放入队列
            myQueue.poll(nums[i - k]);
            //滑动窗口加入最后面的元素
            myQueue.add(nums[i]);
            //记录对应的最大值
            res[num++] = myQueue.peek();
        }
        return res;

    }

    // 前k个高频元素
    public int[] topKFrequent(int[] nums, int k) {
        HashMap<Integer, Integer> map = new HashMap<>();

        for (int num : nums) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }

        // 将HashMap的entrySet()转换成List
        List<Map.Entry<Integer, Integer>> list = new ArrayList(map.entrySet());
        // 使用Collections.sort()排序
        Collections.sort(list, new Comparator<Map.Entry<Integer, Integer>>() {
            @Override
            public int compare(Map.Entry<Integer, Integer> o1, Map.Entry<Integer, Integer> o2) {
                // 按照value值升序排列
                return o2.getValue() - o1.getValue();
            }
        });
        // 创建一个新的有序HashMap
        LinkedHashMap<Integer, Integer> sortedMap = new LinkedHashMap<>();
        for (Map.Entry<Integer, Integer> entry : list) {
            sortedMap.put(entry.getKey(), entry.getValue());
        }

        int[] res = new int[k];
        int i = 0;
        for (Map.Entry<Integer, Integer> entry : sortedMap.entrySet()) {
            res[i++] = entry.getKey();
            if (i >= k){
                break;
            }
        }

        return res;
    }


    // 二叉树
    // 前中后序 分别使用递归和迭代法实现
    // 递归实现
    public List<Integer> preorderTraversal(TreeNode root) {
        ArrayList<Integer> res = new ArrayList<>();
        preSearch(root, res);
        return res;
    }

    public void preSearch(TreeNode root, ArrayList<Integer> res){
        if (root == null){
            return;
        }
        res.add(root.val);
        preSearch(root.left, res);
        preSearch(root.right, res);
    }
    // 迭代版本  前序  中序  后序
    // 前序
    public List<Integer> preorderTraversal2(TreeNode root) {
        ArrayList<Integer> res = new ArrayList<>();
        if (root == null){
            return res;
        }

        Stack<TreeNode> nodeStack = new Stack<>();
        nodeStack.push(root);
        while (!nodeStack.isEmpty()){
            TreeNode pop = nodeStack.pop();
            res.add(pop.val); // 将元素放入res数组中

            if (pop.left != null){
                nodeStack.push(pop.left); // 遍历元素
            }
            if (pop.right!=null){
                nodeStack.push(pop.right); // 遍历元素
            }
        }

        return res;
    }
    // 中序
    public List<Integer> inorderTraversal(TreeNode root) {
        ArrayList<Integer> res = new ArrayList<>();

        Stack<TreeNode> nodeStack = new Stack<>();

        TreeNode cur = root;
        while (cur != null || !nodeStack.isEmpty()){ // 当root不为空，栈为空时加入节点，当root为空（遍历到最左边节点时，处理节点）
            if (cur != null){
                nodeStack.push(cur);
                cur = cur.left;  // 一直遍历到左子树节点末端  左
            }else {
                // 开始处理节点
                cur = nodeStack.pop();  // 取出节点  中
                res.add(cur.val);
                // 处理右节点
                cur = cur.right;  // 右

            }
        }
        return res;

    }

    // 后序  在前序遍历上修改，由 中左右 -> 中右左 -> 左右中
    public List<Integer> postorderTraversal(TreeNode root) {
        ArrayList<Integer> res = new ArrayList<>();
        if (root == null){
            return res;
        }

        Stack<TreeNode> nodeStack = new Stack<>();
        nodeStack.push(root);
        while (!nodeStack.isEmpty()){
            TreeNode cur = nodeStack.pop();
            res.add(cur.val);

            if (cur.left != null){
                nodeStack.push(cur.left);
            }
            if (cur.right != null){
                nodeStack.push(cur.right);
            }
        }

        ArrayList<Integer> result = new ArrayList<>();
        for (Integer re : res) {
            result.add(0, re);
        }
        return result;
    }


    // 二叉树的层序遍历  广度优先思想   迭代法
    public List<List<Integer>> levelOrder(TreeNode root) {
        ArrayList<List<Integer>> res = new ArrayList<>();

        if (root == null){
            return res;
        }

        Deque<TreeNode> deque = new LinkedList<>();
        deque.addLast(root);

        while (!deque.isEmpty()){
            ArrayList<Integer> temp = new ArrayList<>();
            int len = deque.size();
            for (int i = 0; i < len; i++) {
                TreeNode cur = deque.pollFirst();
                temp.add(cur.val);
                if (cur.left != null){
                    deque.addLast(cur.left);
                }
                if (cur.right != null){
                    deque.addLast(cur.right);
                }
            }
            res.add(new ArrayList<>(temp));
        }

        return res;
    }

    // 递归法   深度优先思想
    public List<List<Integer>> resList = new ArrayList<List<Integer>>();

    public List<List<Integer>> levelOrder2(TreeNode root) {
        checkFun01(root,0);
        return resList;
    }

    //DFS--递归方式
    public void checkFun01(TreeNode node, Integer deep) {
        if (node == null) return;
        deep++;

        if (resList.size() < deep) {
            //当层级增加时，list的Item也增加，利用list的索引值进行层级界定     前序遍历
            List<Integer> item = new ArrayList<Integer>();
            resList.add(item);
        }
        resList.get(deep - 1).add(node.val);

        checkFun01(node.left, deep);
        checkFun01(node.right, deep);
    }

    // 翻转二叉树   前序和后序遍历实现 以及层序遍历实现
    // 前序遍历 和后序遍历   中序遍历会翻转两次
    public TreeNode invertTree(TreeNode root) {
        invertTree_pre_rec(root);
        return root;
    }

    public void invertTree_pre_rec(TreeNode root) {
        if (root == null){
            return;
        }
        swap(root);
        invertTree_pre_rec(root.left);
        invertTree_pre_rec(root.right);
//        swap(root);
    }
    public void swap(TreeNode root){
        TreeNode temp = root.right;
        root.right = root.left;
        root.left = temp;
    }


    // 层序遍历 （迭代法）
    public TreeNode invertTree2(TreeNode root) {
        if (root == null) {
            return null;
        }
        ArrayDeque<TreeNode> deque = new ArrayDeque<>();
        deque.offer(root);
        while (!deque.isEmpty()) {
            int size = deque.size();
            while (size-- > 0) {
                TreeNode node = deque.poll();
                swap2(node);
                if (node.left != null) deque.offer(node.left);
                if (node.right != null) deque.offer(node.right);
            }
        }
        return root;
    }

    public void swap2(TreeNode root) {
        TreeNode temp = root.left;
        root.left = root.right;
        root.right = temp;
    }

    // 对称二叉树
    // 中序遍历  判断是否是对称序列       错误解法
    public boolean isSymmetric(TreeNode root) {
        ArrayList<Integer> res = new ArrayList<>();

        Stack<TreeNode> nodeStack = new Stack<>();

        TreeNode cur = root;
        while (cur != null || !nodeStack.isEmpty()){ // 当root不为空，栈为空时加入节点，当root为空（遍历到最左边节点时，处理节点）
            if (cur != null){
                nodeStack.push(cur);
                cur = cur.left;  // 一直遍历到左子树节点末端  左
            }else {
                // 开始处理节点
                cur = nodeStack.pop();  // 取出节点  中
                res.add(cur.val);
                // 处理右节点
                cur = cur.right;  // 右

            }
        }

        int left = 0;
        int right = res.size() - 1;
        while (left < right){
            if (!Objects.equals(res.get(left++), res.get(right--))){
                return false;
            }
        }

        return true;
    }


    boolean compare(TreeNode left, TreeNode right) {
        // 首先排除空节点的情况
        if (left == null && right != null) return false;
        else if (left != null && right == null) return false;
        else if (left == null && right == null) return true;
            // 排除了空节点，再排除数值不相同的情况
        else if (left.val != right.val) return false;

        // 此时就是：左右节点都不为空，且数值相同的情况
        // 此时才做递归，做下一层的判断
        boolean outside = compare(left.left, right.right);   // 左子树：左、 右子树：右
        boolean inside = compare(left.right, right.left);    // 左子树：右、 右子树：左
        boolean isSame = outside && inside;                    // 左子树：中、 右子树：中 （逻辑处理）
        return isSame;

    }
    // 递归写法
    boolean isSymmetric2(TreeNode root) {
        if (root == null) return true;
        return compare(root.left, root.right);
    }


    // 二叉树的最大深度   前序求的是深度，后序求的是高度
    // 后序遍历
    public int maxDepth(TreeNode root) {
        if (root == null) return 0;
        return 1 + Math.max(maxDepth(root.left), maxDepth(root.right));
    }


    // 也可以使用前序遍历（回溯）和层序遍历求最大深度
    // 前序遍历
    public int maxDepth2(TreeNode root) {
        maxDepth_pre(root, 0);
        return result;
    }
    private int result = 0;
    public void maxDepth_pre(TreeNode root, int depth) {

        if (root == null) return;

        depth++;
        result = result < depth ? depth : result; // 更新树的高度
        maxDepth_pre(root.left, depth);
        maxDepth_pre(root.right, depth);
        depth--;

    }

    // 层序遍历
    public int maxDepth3(TreeNode root) {
       if (root == null) return 0;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        int depth = 0;
        while (!queue.isEmpty()){
            int len = queue.size();
            while (len-- > 0){
                TreeNode node = queue.poll();
                if (node.left != null){
                    queue.add(node.left);
                }
                if (node.right != null){
                    queue.add(node.right);
                }
            }
            depth++;
        }

        return depth;
    }


    // 推广：n叉树的最大深度   依然使用递归和迭代法  思路和二叉树一样
    // 递归
    public int maxDepth(NodeN root) {
        if (root == null) return 0;

        int depth = 0;
        for (int i = 0; i < root.children.size(); i++) {
            depth = Math.max(depth, maxDepth(root.children.get(i)));  // 计算root出发 所有孩子节点的最大深度就是该n叉树的最大深度
        }

        return depth + 1;
    }

    // 迭代
    public int maxDepth_cen(NodeN root) {
        if (root == null) return 0;
        Queue<NodeN> queue = new LinkedList<>();
        queue.add(root);
        int depth = 0;
        while (!queue.isEmpty()){
            int len = queue.size();
            while (len-- > 0){
                NodeN node = queue.poll();
                for (int i = 0; i < node.children.size(); i++) {
                    if (node.children.get(i) != null){
                        queue.add(node.children.get(i));
                    }
                }
            }
            depth++;
        }

        return depth;
    }

    // 二叉树的最小深度   注意：这样写会存在误区
    public int minDepth(TreeNode root) {
        if (root == null) return 0;
        return 1 + Math.min(minDepth(root.left), minDepth(root.right));
    }

    // 正确写法
    public int minDepth2(TreeNode root) {
        if (root == null) return 0;

        int left = minDepth2(root.left);
        int right = minDepth2(root.right);

        if (root.left == null && root.right != null){  // 目的是找到叶子节点
            return 1 + right;
        }else if (root.left != null && root.right == null){
            return 1 + left;
        }

        return 1 + Math.min(left, right);
    }

    // 完全二叉树的节点个数
    // 层序遍历
    public int countNodes(TreeNode root) {
        if (root == null) return 0;

        int res = 0;

        ArrayDeque<TreeNode> deque = new ArrayDeque<>();
        deque.offer(root);
        while (!deque.isEmpty()) {
            int size = deque.size();
            while (size-- > 0) {
                TreeNode node = deque.poll();
                res ++;
                if (node.left != null) deque.offer(node.left);
                if (node.right != null) deque.offer(node.right);
            }
        }

        return res;
    }

    // 递归计算个数
    public int countNodes2(TreeNode root) {
        if (root == null) return 0;

        int left = countNodes2(root.left);
        int right = countNodes2(root.right);

        return 1 + left + right;
    }
    // 精简
    public int countNodes3(TreeNode root) {
        if (root == null) return 0;
        return 1 + countNodes3(root.left) + countNodes3(root.right);
    }


    // 平衡二叉树    一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过 1 。  层序遍历 + 判断每个节点的左右子树高度差  （迭代法）
    public boolean isBalanced(TreeNode root) {


        if (root == null) return true;

        ArrayDeque<TreeNode> deque = new ArrayDeque<>();
        deque.offer(root);
        while (!deque.isEmpty()) {
            int size = deque.size();
            while (size-- > 0) {
                TreeNode node = deque.poll();

                if (node.left == null && node.right != null){
                    if (isBalanced_depth(node.right) > 1){
                        return false;
                    }
                }else if (node.left != null && node.right == null){
                    if (isBalanced_depth(node.left) > 1){
                        return false;
                    }
                }else if (node.left != null && node.right != null){
                    if (Math.abs(isBalanced_depth(node.left) - isBalanced_depth(node.right)) > 1){
                        return false;
                    }
                }
                if (node.left != null) deque.offer(node.left);
                if (node.right != null) deque.offer(node.right);
            }
        }

        return true;
    }

    public int isBalanced_depth(TreeNode root){
        if (root == null) return 0;
        return 1 + Math.max(isBalanced_depth(root.left), isBalanced_depth(root.right));
    }


    // 递归法   比迭代法高效
    public boolean isBalanced2(TreeNode root) {
        return isBalanced_depth2(root) != -1;
    }

    public int isBalanced_depth2(TreeNode root){  // 递归求每一个节点高度   后序遍历
        if (root == null){  // 递归终止条件
            return 0;
        }

        int left = isBalanced_depth2(root.left);  // 左
        if (left == -1){   // -1  剪枝
            return -1;
        }
        int right = isBalanced_depth2(root.right);  // 右
        if (right == -1){
            return -1;
        }

        return Math.abs(left - right) > 1? -1: 1 + Math.max(left, right);  // 中
    }



    // 二叉树的所有路径
    public List<String> binaryTreePaths(TreeNode root) {

        List<String> res = new ArrayList<>();
        binaryTreePaths_rec(root, res, "");

        return res;
    }
    public void binaryTreePaths_rec(TreeNode root, List<String> res, String temp) {
        if (root == null){
            return;
        }
        temp += root.val;  // 前序遍历  中
        temp += "->";
        if (root.left == null && root.right == null){ // 叶子节点
            res.add(temp.substring(0, temp.length() - 2));
        }

        binaryTreePaths_rec(root.left, res, temp); // 左
        binaryTreePaths_rec(root.right, res, temp); // 右
    }

    // 左叶子之和
    public int sumOfLeftLeaves(TreeNode root) {
        if (root == null) return 0;
        if (root.left == null && root.right== null) return 0;

        int leftValue = sumOfLeftLeaves(root.left);    // 左
        if (root.left != null && root.left.left == null && root.left.right == null) { // 左子树就是一个左叶子的情况
            leftValue = root.left.val;
        }
        int rightValue = sumOfLeftLeaves(root.right);  // 右

        int sum = leftValue + rightValue;               // 中
        return sum;
    }


    // 找树左下角的值  // 直接层序遍历找到值
    public int findBottomLeftValue(TreeNode root) {

        if (root == null) {
            return 0;
        }
        ArrayList<ArrayList<Integer>> res = new ArrayList<>();
        ArrayDeque<TreeNode> deque = new ArrayDeque<>();
        deque.offer(root);
        while (!deque.isEmpty()) {
            int size = deque.size();
            ArrayList<Integer> temp = new ArrayList<>();
            while (size-- > 0) {
                TreeNode node = deque.poll();
                temp.add(node.val);
                if (node.left != null) deque.offer(node.left);
                if (node.right != null) deque.offer(node.right);
            }
            res.add(temp);
        }
        return res.get(res.size() - 1).get(0);
    }

    // 或者使用前序递归遍历   最大深度 + 最左边叶子节点
    int maxDepth = Integer.MIN_VALUE;
    int result_;
    void traversal(TreeNode root, int depth) {
        if (root.left == null && root.right == null) {
            if (depth > maxDepth) {
                maxDepth = depth;
                result_ = root.val;
            }
            return;
        }
        if (root.left != null) {
            depth++;
            traversal(root.left, depth);
            depth--; // 回溯
        }
        if (root.right != null) {
            depth++;
            traversal(root.right, depth);
            depth--; // 回溯
        }
        return;
    }
    public int findBottomLeftValue2(TreeNode root) {
        traversal(root, 0);
        return result_;
    }

    // 路径总和
    boolean res_hasPathSum = false;
    public boolean hasPathSum(TreeNode root, int targetSum) {
        hasPathSum_rec(root, targetSum, 0);
        return res_hasPathSum;
    }



    public void hasPathSum_rec(TreeNode root, int targetSum, int sum) {
        if (root == null){
            return;
        }
        sum += root.val;  // 前序遍历  中
        if (root.left == null && root.right == null){ // 叶子节点
            if (sum == targetSum){
                res_hasPathSum = true;
            }
        }

        hasPathSum_rec(root.left, targetSum, sum); // 左
        hasPathSum_rec(root.right, targetSum, sum); // 右

    }

    // 中序和后序 构建二叉树
    public int findIndex(int[] targetArr, int target){
        for (int i = 0; i < targetArr.length; i++) {
            if (targetArr[i] == target){
                return i;
            }
        }
        return -1;
    }
    // 数组切片
    public int[] sliceArr(int[] targetSliceArr, int start, int end){  // 左闭右闭
        int[] res = new int[end - start + 1];
        for (int i = start; i <= end; i++) {
            res[i - start] = targetSliceArr[i];
        }
        return res;
    }
    public TreeNode buildTree(int[] inorder, int[] postorder) {
        if (postorder.length == 0) return null;

        // 1、利用后序遍历的最后一个节点为根节点
        TreeNode root = new TreeNode(postorder[postorder.length - 1]);
        // 2、在中序遍历里找到根节点位置
        int index = findIndex(inorder, postorder[postorder.length - 1]);
        System.out.println(index);
        // 3、将中序遍历数组按照根节点位置分成左右子树
        int[] left = sliceArr(inorder, 0, index - 1);
        int[] right = sliceArr(inorder, index + 1, inorder.length - 1);
        // 4、将后序遍历数组按照左右子树区间位置切割
        int[] leftPost = sliceArr(postorder, 0, left.length - 1);
        int[] rightPost = sliceArr(postorder, left.length, left.length + right.length - 1);
        // 5、递归构建二叉树
        root.left = buildTree(left, leftPost);
        root.right = buildTree(right, rightPost);
        return root;
    }

    // 最大二叉树
    public TreeNode constructMaximumBinaryTree(int[] nums) {

        if (nums.length == 0) return null;

        int maxValue = Arrays.stream(nums).max().getAsInt();
        TreeNode root = new TreeNode(maxValue);

        int index = findIndex(nums, maxValue); // 左闭右闭

        int[] left = sliceArr(nums, 0, index - 1);
        int[] right = sliceArr(nums, index + 1, nums.length - 1);


        root.left = constructMaximumBinaryTree(left);
        root.right = constructMaximumBinaryTree(right);

        return root;
    }


    // 合并二叉树
    public TreeNode mergeTrees(TreeNode root1, TreeNode root2) {
        // 前序遍历
        if (root1 == null && root2 != null){
            return root2;
        }else if (root1 != null && root2 == null){
            return root1;
        }else if (root1 == null && root2 == null){
            return null;
        }
        TreeNode root = new TreeNode(root1.val + root2.val);

        root.left = mergeTrees(root1.left , root2.left);
        root.right = mergeTrees(root1.right, root2.right);

        return root;
    }

    // 二叉树中的搜索
    TreeNode root_searchBST = null;
    public TreeNode searchBST(TreeNode root, int val) {
        // 前序遍历
        searchBST_rec(root, val);
        return root_searchBST;
    }
    public void searchBST_rec(TreeNode root, int val) {
        // 前序遍历
        if (root == null) return;


        if (root.val == val){
            root_searchBST = root;
        }

        searchBST_rec(root.left, val);
        searchBST_rec(root.right, val);
    }

    // 验证二叉搜索树
    public boolean isValidBST(TreeNode root) {
        // 直接中序遍历一下，，再对得到数组排序，，依次验证是否升序排序
        ArrayList<Integer> res = new ArrayList<>();
        isValidBST_inorder(root, res);

//        ArrayList<Integer> res2 = new ArrayList<>(res);  // 多此一举
//        Collections.sort(res2);
//
//        HashSet<Integer> set = new HashSet<>(res2);
//        if (set.size() != res2.size()){
//            return false;
//        }
//
//        for (int i = 0; i < res.size(); i++) {
//            if (!Objects.equals(res.get(i), res2.get(i))){
//                return false;
//            }
//        }
        for (int i = 1; i < res.size(); i++) {
            if (res.get(i) <= res.get(i - 1)){
                return false;
            }
        }


        return true;
    }

    public void isValidBST_inorder(TreeNode root, ArrayList<Integer> res) {
        if (root == null){
            return;
        }

        isValidBST_inorder(root.left, res);
        res.add(root.val);
        isValidBST_inorder(root.right, res);

    }

    // 也可以使用迭代法实现（保存pre和cur指针）

    // 最小绝对值差值
    public int getMinimumDifference(TreeNode root) {

        ArrayList<Integer> res = new ArrayList<>();
        getMinimumDifference_inorder(root, res);

        int minValue = Integer.MAX_VALUE;
        for (int i = 1; i < res.size(); i++) {
            minValue = Math.min(Math.abs(res.get(i) - res.get(i - 1)), minValue);
        }

        return minValue;
    }

    public void getMinimumDifference_inorder(TreeNode root, ArrayList<Integer> res) {
        if (root == null){
            return;
        }

        getMinimumDifference_inorder(root.left, res);
        res.add(root.val);
        getMinimumDifference_inorder(root.right, res);

    }

    // 也可以使用中序遍历 或 迭代法 +   前后指针实现


    // 二叉搜索树的最近公共祖先   （1）可以使用二叉树最近公共祖先的思路（从下到上）  （2）也可以利用二叉搜索树的特点的思路（从上到下，查找p、q分为位于root两边）
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
//        （1）
        return lowestCommonAncestor_dfs(root, p, q);
    }


    public TreeNode lowestCommonAncestor_dfs(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || root == p || root == q){
            return root;
        }

        TreeNode left = lowestCommonAncestor_dfs(root.left, p, q);
        TreeNode right = lowestCommonAncestor_dfs(root.right, p, q);

        if (left != null && right != null){
            return root;
        }else if (left == null){
            return right;
        }
        return left;
    }

    public TreeNode lowestCommonAncestor2(TreeNode root, TreeNode p, TreeNode q) {
//        （2）
        return lowestCommonAncestor_dfs2(root, p, q);
    }


    public TreeNode lowestCommonAncestor_dfs2(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null){
            return root;
        }

        if (root.val > p.val && root.val > q.val){
            TreeNode left = lowestCommonAncestor_dfs2(root.left, p, q);
            if (left != null) {
                return left;
            }
        }


        if (root.val < p.val && root.val < q.val){
            TreeNode right = lowestCommonAncestor_dfs2(root.right, p, q);
            if (right != null) {
                return right;
            }
        }

        return root;
    }


    // 二叉搜索树中的插入操作
    public TreeNode insertIntoBST(TreeNode root, int val) {
        return insertIntoBST_dfs(root, val);
    }


    public TreeNode insertIntoBST_dfs(TreeNode root, int val) {
        if (root == null){
            return new TreeNode(val);
        }

        if (root.val > val){
            root.left = insertIntoBST_dfs(root.left, val);
        }

        if (root.val < val){
            root.right = insertIntoBST_dfs(root.right, val);
        }

        return root;
    }

    // 删除二叉搜索树中的节点
    public TreeNode deleteNode(TreeNode root, int key) {
        return deleteNode_dfs(root, key);
    }


    public TreeNode deleteNode_dfs(TreeNode root, int val) {
        if (root == null){
            return root;
        }

        if (root.val == val){

            if (root.left == null){
                return root.right;
            }else if (root.right == null){
                return root.left;
            }else {
                TreeNode cur = root.right;
                while (cur.left != null){
                    cur = cur.left;
                }
                cur.left = root.left;
                root = root.right;


                return root;
            }

        }

        if (root.val > val){
            root.left = deleteNode_dfs(root.left, val);
        }
        if (root.val < val){
            root.right = deleteNode_dfs(root.right, val);
        }

        return root;
    }

    //    修剪二叉搜索树???
    TreeNode trimBST(TreeNode root, int low, int high) {
        if (root == null ) return null;
        if (root.val < low) {
            TreeNode right = trimBST(root.right, low, high); // 寻找符合区间[low, high]的节点
            return right;
        }
        if (root.val > high) {
            TreeNode left = trimBST(root.left, low, high); // 寻找符合区间[low, high]的节点
            return left;
        }
        root.left = trimBST(root.left, low, high); // root->left接入符合条件的左孩子
        root.right = trimBST(root.right, low, high); // root->right接入符合条件的右孩子
        return root;
    }



    public int[] sliceArray(int[] nums, int start, int end){
        int[] res = new int[end - start + 1];  // 左闭右闭
        if (end - start + 1 >= 0) System.arraycopy(nums, start, res, 0, end - start + 1);
        return res;
    }


    // 将有序数组转化成二叉搜索树
    public TreeNode sortedArrayToBST(int[] nums) {

        if (nums.length == 0){
            return null;
        }

        int midValue = nums[nums.length / 2];

        TreeNode root = new TreeNode(midValue);

        root.left = sortedArrayToBST(sliceArray(nums, 0, nums.length / 2 - 1));
        root.right = sortedArrayToBST(sliceArray(nums, nums.length / 2 + 1, nums.length - 1));

        return root;
    }



    // 把二叉搜索树转换成累加树
    public TreeNode convertBST(TreeNode root) {
        convertBST_dfs(root);
        return root;
    }

    // 每个节点的右子树都是大于该节点的值的，所以累加该点的值 = 遍历并累加其右子树的所有节点的值

    // 反中序遍历
    int pre = 0;
    public void convertBST_dfs(TreeNode root) {

        if (root == null) {
            return;
        }

        convertBST_dfs(root.right);
        root.val += pre;
        pre = root.val;
        convertBST_dfs(root.left);

    }

    // 回溯
    // 组合
    public List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> res = new ArrayList<>();
        ArrayList<Integer> temp = new ArrayList<>();
        combine_dfs(n, k, temp, res, 1);
        return res;
    }

    // 回溯  ->  n 叉树
    public void combine_dfs(int n, int k, List<Integer> temp, List<List<Integer>> res, int index) {
        if (temp.size() == k){
            res.add(new ArrayList<>(temp));
            return;
        }

        for (int i = index; i <= n; i++) {
            temp.add(i);
            combine_dfs(n, k, temp, res, i+1);
            temp.remove(temp.size() - 1);
        }
    }
    // 剪枝     i <= n - (k - temp.size()) + 1


    // 组合综合III
    public List<List<Integer>> combinationSum3(int k, int n) {


        List<List<Integer>> res = new ArrayList<>();
        ArrayList<Integer> temp = new ArrayList<>();
        combinationSum3_dfs(k, temp, res, 1, 0, n);
        return res;
    }

    public void combinationSum3_dfs(int k, List<Integer> temp, List<List<Integer>> res, int index, int sum, int n) {
        if (sum == n && temp.size() == k){
            res.add(new ArrayList<>(temp));
            return;
        }

        for (int i = index; i <= 9; i++) {
            temp.add(i);
            sum += i;
            // 剪枝：if (sum > targetSum) { // 剪枝操作
            //        sum -= i; // 剪枝之前先把回溯做了
            //        path.pop_back(); // 剪枝之前先把回溯做了
            //        return;
            //    }
            combinationSum3_dfs(k, temp, res, i+1, sum, n);
            temp.remove(temp.size() - 1);
            sum -= i;
        }
    }

    // 分割回文串  回溯法
    public List<List<String>> partition(String s) {

        List<List<String>> res = new ArrayList<>();
        List<String> temp2 = new ArrayList<>();
        partition_dfs(s, res, temp2, 0);
        return res;
    }

    public void partition_dfs(String s, List<List<String>> res, List<String> temp2, int endIndex) {
        if (endIndex >= s.length()){
            res.add(new ArrayList<>(temp2));
            return;
        }


        for (int i = endIndex; i < s.length(); i++) {
            if (judge(s.substring(endIndex, i + 1))){  // 判断是否回文  [endIndex, i] 这个区间就是分割后的子串
                temp2.add(s.substring(endIndex, i + 1));
            }else {
                continue;
            }
            partition_dfs(s, res, temp2, i + 1); // i + 1 保证不被重复选取
            temp2.remove(temp2.size() - 1);
        }
    }

    // 判断回文
    private boolean judge(String arr){

        int left = 0;
        int right = arr.length() - 1;
        while (left < right){
            if (arr.charAt(left) != arr.charAt(right)){
                return false;
            }
            left++;
            right--;
        }

        return true;
    }

    // 复原IP地址  思路其实和上一道题 判断回文一样
    public List<String> restoreIpAddresses(String s) {

        List<List<String>> res = new ArrayList<>();
        List<String> temp2 = new ArrayList<>();
        restoreIpAddresses_dfs(s, res, temp2, 0);

        List<String> result = new ArrayList<>();

        for (List<String> re : res) {    // 太耗时
            StringBuilder ss = new StringBuilder();
            for (int i1 = 0; i1 < re.size(); i1++) {
                ss.append(re.get(i1));
                if (i1 != re.size() - 1) {
                    ss.append(".");
                }
            }
            result.add(ss.toString());
        }


        return result;
    }

    public void restoreIpAddresses_dfs(String s, List<List<String>> res, List<String> temp2, int startIndex) {
        if (startIndex >= s.length() && temp2.size()  == 4){
            res.add(new ArrayList<>(temp2));
            return;
        }

        for (int i = startIndex; i < s.length(); i++) {
            // 这里可以剪枝,当需要判断的子串长度大于3位，就不用判断了
            if (i + 1 -startIndex > 3){
                continue;
            }

            if (judge2(s.substring(startIndex, i + 1))){  // 判断是否满足条件
                temp2.add(s.substring(startIndex, i + 1));
            }else {
                continue;
            }
            restoreIpAddresses_dfs(s, res, temp2, i + 1); // i + 1 保证不被重复选取
            temp2.remove(temp2.size() - 1);
        }

    }

    private boolean judge2(String arr){
        long parseLong;
        try {
            parseLong  = Long.parseLong(arr); // 当出现  9999999999999999999 超过 long范围时会抛出异常：java.lang.NumberFormatException
                                              // 使用 try...catch... 捕获异常
        }catch (Exception e){
            return false;
        }

        // 范围在 0 ~ 255之间  不能有前导的 0
        return parseLong <= 255 && (arr.length() <= 1 || arr.charAt(0) != '0');

    }

    // 子集
    public List<List<Integer>> subsets(int[] nums) {

        List<List<Integer>> res = new ArrayList<>();
        ArrayList<Integer> temp = new ArrayList<>();
        subsets_dfs(nums, res, temp, 0);

        return res;
    }


    public void subsets_dfs(int[] nums, List<List<Integer>> res, List<Integer> temp, int startIndex) {
        res.add(new ArrayList<>(temp));

        if (startIndex >= nums.length){
            return;
        }


        for (int i = startIndex; i < nums.length; i++) {
            temp.add(nums[i]);
            subsets_dfs(nums, res, temp, i+1);
            temp.remove(temp.size() - 1);
        }


    }

    // 子集II   在子集I里增加了一个因素 ------ 元素可以重复
    public List<List<Integer>> subsetsWithDup(int[] nums) {

        List<List<Integer>> res = new ArrayList<>();
        ArrayList<Integer> temp = new ArrayList<>();
        boolean[] used = new boolean[nums.length];
        Arrays.sort(nums);
        subsetsWithDup_dfs(nums, res, temp, 0, used);

        return res;
    }

    public void subsetsWithDup_dfs(int[] nums, List<List<Integer>> res, List<Integer> temp, int startIndex, boolean[] used) {
        res.add(new ArrayList<>(temp));


        for (int i = startIndex; i < nums.length; i++) {
            // 增加条件  used[i - 1] == false 表示同一树层上存在重复元素（不可选取）， used[i - 1] == true  表示同一树枝上存在重复元素（可以选取）
            if (i > 0 && !used[i - 1] && nums[i] == nums[i - 1]){
                continue;
            }
            used[i] = true;
            temp.add(nums[i]);
            subsetsWithDup_dfs(nums, res, temp, i+1, used);
            temp.remove(temp.size() - 1);
            used[i] = false;
        }

    }

    // 递增子序列  回溯法  子序列：不用连续  子串：需要连续          最小长度为 2
    public List<List<Integer>> findSubsequences(int[] nums) {


        List<List<Integer>> res = new ArrayList<>();
        ArrayList<Integer> temp = new ArrayList<>();
        findSubsequences_dfs(nums, res, temp, 0);

        return res;
    }


    public void findSubsequences_dfs(int[] nums, List<List<Integer>> res, List<Integer> temp, int startIndex) {
        // 加入判断
        if (temp.size() > 1){
            res.add(new ArrayList<>(temp));
        }
        int[] used = new int[201]; // 使用数组充当哈希表 对本层元素进行去重
        for (int i = startIndex; i < nums.length; i++) {
            // 增加条件  used[i - 1] == false 表示同一树层上存在重复元素（不可选取）， used[i - 1] == true  表示同一树枝上存在重复元素（可以选取）
            if ((temp.size() > 0 && nums[i] < temp.get(temp.size() - 1)) || used[nums[i] + 100] == 1){
                continue;
            }
            used[nums[i] + 100] = 1;
            temp.add(nums[i]);
            findSubsequences_dfs(nums, res, temp, i+1);
            temp.remove(temp.size() - 1);
        }

    }

    // 全排列
    public List<List<Integer>> permute(int[] nums) {

        List<List<Integer>> res = new ArrayList<>();
        ArrayList<Integer> temp = new ArrayList<>();
        boolean[] used = new boolean[nums.length];
        permute_dfs(nums, res, temp, used);

        return res;
    }

    public void permute_dfs(int[] nums, List<List<Integer>> res, List<Integer> temp, boolean[] used) {
        if (temp.size() == nums.length){
            res.add(new ArrayList<>(temp));
            return;
        }


        for (int i = 0; i < nums.length; i++) {
            if (used[i]){
                continue;
            }
            used[i] = true;
            temp.add(nums[i]);
            permute_dfs(nums, res, temp, used);
            temp.remove(temp.size() - 1);
            used[i] = false;
        }


    }

    // 全排列II    遇上一道题的不同之处在于 ----- 元素可以重复
    public List<List<Integer>> permuteUnique(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        ArrayList<Integer> temp = new ArrayList<>();
        boolean[] used = new boolean[nums.length];
        Arrays.sort(nums);
        permuteUnique_dfs(nums, res, temp, used);

        return res;


    }


    public void permuteUnique_dfs(int[] nums, List<List<Integer>> res, List<Integer> temp, boolean[] used) {

        if (temp.size() == nums.length){
            res.add(new ArrayList<>(temp));
            return;
        }


        for (int i = 0; i < nums.length; i++) {
            if (i > 0 && nums[i] == nums[i - 1] && !used[i - 1]){
                continue;
            }
            if (!used[i]){
                used[i] = true;
                temp.add(nums[i]);
                permuteUnique_dfs(nums, res, temp, used);
                temp.remove(temp.size() - 1);
                used[i] = false;
            }

        }

    }

    // 贪心  从局部最优到全局最优

    // 分发饼干
    public int findContentChildren(int[] g, int[] s) {

        Arrays.sort(g);  // 胃口
        Arrays.sort(s); // 饼干
        int i = 0;
        int j = 0;
        while (i < s.length && j < g.length) {
            if (g[j] <= s[i++]){
                j++;
            }
        }

        return j;
    }

    // 摆动序列
    public int wiggleMaxLength(int[] nums) {
        if (nums.length <= 1) {
            return nums.length;
        }
        //当前差值
        int curDiff = 0;
        //上一个差值
        int preDiff = 0;
        int count = 1;
        for (int i = 1; i < nums.length; i++) {
            //得到当前差值
            curDiff = nums[i] - nums[i - 1];
            //如果当前差值和上一个差值为一正一负
            //等于0的情况表示初始时的preDiff
            if ((curDiff > 0 && preDiff <= 0) || (curDiff < 0 && preDiff >= 0)) {
                count++;
                preDiff = curDiff;
            }
        }
        return count;
    }

    // 最大子数组和   贪心或者动规
    // 贪心
    public int maxSubArray(int[] nums) {

        int result = Integer.MIN_VALUE;
        int count = 0;
        for (int i = 0; i < nums.length; i++) {
            count += nums[i];
            if (count > result) { // 取区间累计的最大值（相当于不断确定最大子序终止位置）
                result = count;
            }
            if (count <= 0) count = 0; // 相当于重置最大子序起始位置，因为遇到负数一定是拉低总和
        }
        return result;
    }

    // 买卖股票的最佳时机II
    public int maxProfit(int[] prices) {

        int res = 0;

        for (int i = 1; i < prices.length; i++) {
            res += Math.max(prices[i] - prices[i - 1], 0);
        }

        return res;
    }

    // 跳跃游戏
    public boolean canJump(int[] nums) {

        int cover = 0; // 所在位置索引
        if (nums.length == 1) return true;
        for (int i = 0; i <= cover; i++) {
            cover = Math.max(cover, nums[i] + i);
            if (cover >= nums.length - 1) return true;
        }

        return false;
    }

    // 跳跃游戏II    到达终点的最小跳跃次数
    public int jump(int[] nums) {
        int curCover = 0; // 当前覆盖最大范围
        int count = 0;
        int nextCover = 0; // 下一步覆盖最大范围
        for (int i = 0; i < nums.length - 1; i++) {  // 其精髓在于控制移动下标i只移动到nums.size() - 2的位置，
                                                     // 所以移动下标只要遇到当前覆盖最远距离的下标，直接步数加一，不用考虑别的了。
            nextCover = Math.max(nextCover, nums[i] + i);
            if (i == curCover) {
                curCover = nextCover;
                count ++;
            }
        }

        return count;
    }

    // k次取反后最大化的数组和
    public int largestSumAfterKNegations(int[] nums, int k) {
        // 按绝对值从大到小排列
        nums = IntStream.of(nums)
                .boxed()
                .sorted((o1, o2) -> Math.abs(o2) - Math.abs(o1))
                .mapToInt(Integer::intValue).toArray();

        for (int i = 0; i < nums.length; i++) {
            if (nums[i] < 0 && k > 0) {
                nums[i] = -nums[i];
                k--;
            }
        }

        if (k % 2 == 1){
            nums[nums.length - 1] = -nums[nums.length - 1];
        }

        return Arrays.stream(nums).sum();
    }


    // 加油站
    public int canCompleteCircuit(int[] gas, int[] cost) {
        int curSum = 0;
        int allSum = 0;
        int start = 0;
        for (int i = 0; i < gas.length; i++) {
            curSum += gas[i] - cost[i];
            allSum += gas[i] - cost[i];

            if (curSum < 0){
                start = i + 1;
                curSum = 0;
            }
        }

        if (allSum < 0){
            return -1;
        }

        return start;
    }


    // 柠檬水找零
    public boolean lemonadeChange(int[] bills) {
        // 优先使用 10

        int five = 0;
        int ten = 0;
        for (int bill : bills) {
            if (bill == 5) {
                five++;
            } else if (bill == 10) {
                if (five > 0) {
                    five--;
                    ten++;
                } else {
                    return false;
                }
            } else {
                if (five > 0 && ten > 0) {
                    five--;
                    ten--;
                } else if (five >= 3) {
                    five -= 3;
                } else {
                    return false;
                }
            }
        }
        return true;
    }

    // 根据身高重建队列
    public int[][] reconstructQueue(int[][] people) {
        Arrays.sort(people, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                if (o1[0] != o2[0]){
                    return o2[0] - o1[0];
                }else {
                    return o1[1] - o2[1];
                }
            }
        });

        List<int[]> list = new LinkedList<>();

        for (int[] person : people) {
            list.add(Math.min(person[1], list.size()), person);
        }

        return list.toArray(new int[list.size()][]);
    }

    // 用最小数量的箭射爆气球
    public int findMinArrowShots(int[][] points) {

        Arrays.sort(points, ((o1, o2) -> {
            return Integer.compare(o1[0], o2[0]);
        }));
        int count = 1;
        for (int i = 1; i < points.length; i++) {

            if (points[i][0] > points[i - 1][1]){
                count ++;  // 不挨着一定需要一支箭
            }else {
                points[i][1] = Math.min(points[i - 1][1], points[i][1]);  // 重叠则更新右边界，取最小
            }
        }
        return count;
    }

    // 无重叠区间
    public int eraseOverlapIntervals(int[][] intervals) {

        if (intervals.length == 0) return 0;

        int count = 0; // 记录非交叉区间个数
        Arrays.sort(intervals, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                if (o1[1] != o2[1]){
                    return o1[1] - o2[1]; // 右边界从小到大排序
                }else {
                    return o2[0] - o1[0];
                }

            }
        });

        int end = intervals[0][1];

        for (int i = 1; i < intervals.length; i++) {
            if (end <= intervals[i][0]){
                end = intervals[i][1]; // 跟新右边界
                count++;
            }
        }

        return intervals.length - count - 1;
    }


    // 划分字母区间
    public List<Integer> partitionLabels(String s) {

        int[] record = new int[27];
        for (int i = 0; i < s.length(); i++) {
            record[s.charAt(i) - '0'] = i;
        }

        List<Integer> res = new LinkedList<>();
        int maxIndex = 0;
        int left = 0;
        int right = 0;
        for (int i = 0; i < s.length(); i++) {
            maxIndex = Math.max(maxIndex, record[s.charAt(i) - '0']);
            if (i == maxIndex) {
                res.add(right - left + 1);
                left = i + 1;
            }
            right++;
        }

        return res;
    }

    // 合并区间
    public int[][] merge(int[][] intervals) {

        Arrays.sort(intervals, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[0] - o2[0];  // 左边界从小到大排序
            }
        });

        ArrayList<ArrayList<Integer>> res = new ArrayList<>();

        int end = intervals[0][1]; // 初始右边界
        int start = intervals[0][0]; // 初始左边界
        for (int i = 1; i < intervals.length; i++) {
            if (end >= intervals[i][0]){ // 重叠
                end = Math.max(end, intervals[i][1]);
            }else { // 不重叠
                ArrayList<Integer> temp = new ArrayList<>();
                temp.add(start);
                temp.add(end);
                res.add(temp);
                // 更新start
                start = intervals[i][0];
                // 更新end
                end = Math.max(end, intervals[i][1]);
            }
        }
        ArrayList<Integer> temp = new ArrayList<>();
        temp.add(start);
        temp.add(end);
        res.add(temp); // 最后还有一次
        int[][] result = new int[res.size()][2];
        int i = 0;
        for (ArrayList<Integer> re : res) {
            result[i][0] = re.get(0);
            result[i][1] = re.get(1);
            i++;
        }

        return result;
    }

    // 单调递增的数字
    public int monotoneIncreasingDigits(int n) {

        String[] split = (n + "").split("");

        int start = split.length;
        for (int i = start - 1; i > 0; i--) {
            if (split[i - 1].charAt(0) - '0' > split[i].charAt(0) - '0'){
                split[i - 1] = (Integer.parseInt(split[i - 1]) - 1) + ""; // 减一     332  ->  322  ->  212
                start = i;
            }

        }

        for (int i = start; i < split.length; i++) {
            split[i] = "9";  // 更新为9       -> 299
        }

        return Integer.parseInt(String.join("", split));
    }


    // 单调栈
    // 每日温度
    // 栈底  --->  栈顶
    // 查找右边第一个比自己大的元素，使用单调递增栈
    // 查找右边第一个比自己小的元素，使用单调递减栈
    public int[] dailyTemperatures(int[] temperatures) {

        Deque<Integer> queue = new LinkedList<>();
        int[] result = new int[temperatures.length];
        for (int i = 0; i < temperatures.length; i++) {
            while (!queue.isEmpty() && temperatures[i] > temperatures[queue.peek()]){
                result[queue.peek()] = i - queue.peek();
                queue.pop();
            }
            queue.push(i);

        }

        return result;
    }

    // 下一个更大元素I
    public int[] nextGreaterElement(int[] nums1, int[] nums2) {

        Deque<Integer> queue = new LinkedList<>();
        int[] result = new int[nums2.length];
        for (int i = 0; i < nums2.length; i++) {
            while (!queue.isEmpty() && nums2[i] > nums2[queue.peek()]){
                result[queue.peek()] = i; // 右边第一个比该元素大的元素的索引
                queue.pop();
            }
            queue.push(i);

        }

        int[] res = new int[nums1.length];
        int index = 0;
        for (int i : nums1) {
            for (int i1 = 0; i1 < nums2.length; i1++) {
                if (i == nums2[i1]){
                    if(result[i1] == 0){
                        res[index++] = -1;
                    }else{
                        res[index++] = nums2[result[i1]];
                    }
                }
            }
        }

        return res;
    }


    // 下一个更大的元素II   环形查找   只需在 每日温度上遍历两遍数组
    public int[] nextGreaterElements(int[] nums) {

        Deque<Integer> queue = new LinkedList<>();
        int[] result = new int[nums.length];
        Arrays.fill(result, -1);
        for (int i = 0; i < nums.length * 2; i++) {
            while (!queue.isEmpty() && nums[i % nums.length] > nums[queue.peek()]){
                result[queue.peek()] = nums[i % nums.length];
                queue.pop();
            }
            queue.push(i % nums.length);

        }

        return result;

    }





    // 动规
    public int maxSubArray2(int[] nums) {

        int pre = 0;
        int maxAns = nums[0];
        for (int x : nums) {
            pre = Math.max(pre + x, x);
            maxAns = Math.max(maxAns, pre);
        }
        return maxAns;

    }

    // 最后一块石头的重量II
    public int lastStoneWeightII(int[] stones) {
        // dp[i]表示容量为i能装的最大重量
        int[] dp = new int[15000];
        int v = 0;
        for (int value: stones){
            v+=value;
        }
        v/=2;

        for (int i = 0; i < stones.length; i++) {
            for (int j = v; j >= stones[i]; j--) {
                dp[j] = Math.max(dp[j], dp[j - stones[i]] + stones[i]);
            }
        }


        return Arrays.stream(stones).sum() - dp[v] - dp[v];
    }
    // 目标和
    public int findTargetSumWays(int[] nums, int target) {

        // dp[i]  表示 填满i的背包有几种组合   加法总和(x)为 + 减法总和(sum - x)也为 +
        // 所以：target = x - (sum - x) => target = (target + sum) / 2
        // 所以：背包容量 v = (target + sum) / 2
        int sum = 0;
        for (int value: nums){
            sum+=value;
        }
        if ( target < 0 && sum < -target) return 0;
        if (target > Math.abs(sum)){
            return 0;
        }

        int v = (target + sum) / 2;
        if(v < 0) v = -v;
        if (v % 2 == 1){
            return 0;
        }

        int[] dp = new int[v + 1];
        dp[0] = 1;
        for (int num : nums) {
            for (int j = v; j >= num; j--) {
                dp[j] += dp[j - num];
            }
        }

        return dp[v];
    }
    // 一和零  m 个 0 ， n 个 1   两个维度的 01 背包
    public int findMaxForm(String[] strs, int m, int n) {

        int[][] dp = new int[m+1][n+1]; // 表示 最多 i 个 0 和 最多 j 个 1组成的最长子串

        for (String str: strs){
            int zeroNum = 0;
            int oneNum = 0;
            for (char c: str.toCharArray()){
                if (c == '0'){
                    zeroNum ++;
                }else {
                    oneNum ++;
                }
            }

            for (int i = m; i >= zeroNum; i--) {
                for (int j = n; j >= oneNum; j--) {
                    dp[i][j] = Math.max(dp[i][j], dp[i - zeroNum][j - oneNum] + 1);
                }
            }
        }

        return dp[m][n];
    }

    // 零钱兑换II
    public int change(int amount, int[] coins) {

        int[] dp = new int[amount + 1];
        dp[0] = 1;
        for (int coin : coins) {
            for (int i = 1; i <= amount; i++) {
                if (coin <= i) {
                    dp[i] += dp[i - coin];
                }
            }
        }

        return dp[amount];
    }


    // 组合总和IV        本题题目描述说是求组合，但又说是可以元素相同顺序不同的组合算两个组合，其实就是求排列！
//    如果求组合数就是外层for循环遍历物品，内层for遍历背包。
//    如果求排列数就是外层for遍历背包，内层for循环遍历物品。
    public int combinationSum4(int[] nums, int target) {
        int[] dp = new int[target + 1];
        dp[0] = 1;
        for (int i = 1; i <= target; i++) {
            for (int num : nums) {
                if (num <= i){
                    dp[i] += dp[i - num];
                }
            }
        }

        return dp[target];
    }


    // 零钱兑换
    public int coinChange(int[] coins, int amount) {
        int[] dp = new int[amount + 1]; // 表示凑成 i 元需要的最小个数

        Arrays.fill(dp, Integer.MAX_VALUE);
        dp[0] = 0;
        for (int i = 1; i <= amount; i++) {
            for (int coin : coins) {
                if (coin <= i && dp[i - coin] != Integer.MAX_VALUE) {
                    dp[i] = Math.min(dp[i], dp[i - coin] + 1);
                }
            }
        }
        return dp[amount] == Integer.MAX_VALUE ? -1 : dp[amount];
    }


    // 不相交的线    即 最长公共子序列的长度
    public int maxUncrossedLines(int[] nums1, int[] nums2) {

        int[][] dp = new int[nums1.length + 1][nums2.length + 1];

        for (int i = 1; i <= nums1.length; i++) {
            for (int j = 1; j <= nums2.length; j++) {
                if (nums1[i - 1] == nums2[j - 1]){
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                }else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }


        return dp[nums1.length][nums2.length];
    }

    // 最大子数组和
    public int maxSubArray3(int[] nums) {
        int pre = 0;
        int res = nums[0];

        for (int v: nums){
            pre = Math.max(v, pre + v);
            res = Math.max(pre, res);
        }
        return res;

    }

    // 判断子序列
    public boolean isSubsequence(String s, String t) {

        int[][] dp = new int[s.length() + 1][t.length() + 1];

        for (int i = 1; i <= s.length(); i++) {
            for (int j = 1; j <= t.length(); j++) {
                if (s.charAt(i - 1) == t.charAt(j - 1)){
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                }else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }

        return dp[s.length()][t.length()] == s.length();
    }

    // 不同的子序列
    public int numDistinct(String s, String t) {

        int[][] dp = new int[s.length() + 1][t.length() + 1];
        for (int i = 0; i < s.length() + 1; i++) {
            dp[i][0] = 1;
        }

        for (int i = 1; i < s.length() + 1; i++) {
            for (int j = 1; j < t.length() + 1; j++) {
                if (s.charAt(i - 1) == t.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j];
                }else{
                    dp[i][j] = dp[i - 1][j];
                }
            }
        }

        return dp[s.length()][t.length()];
    }

    // 两个字符串的删除操作
    public int minDistance(String word1, String word2) {
        int[][] dp = new int[word1.length() + 1][word2.length() + 1];
        for (int i = 0; i <= word1.length(); i++) dp[i][0] = i;
        for (int j = 0; j <= word2.length(); j++) dp[0][j] = j;

        for (int i = 1; i < word1.length() + 1; i++) {
            for (int j = 1; j < word2.length() + 1; j++) {
                if (word1.charAt(i - 1) != word2.charAt(j - 1)) {
                    dp[i][j] = Math.min(dp[i - 1][j] + 1, Math.min(dp[i][j - 1] + 1, dp[i - 1][j - 1] + 2));
                }else {
                    dp[i][j] = dp[i - 1][j - 1];
                }
            }
        }

        return dp[word1.length()][word2.length()];
    }

    // 最长回文子串
    public int countSubstrings(String s) {

        if (s.length() == 1) return 1;

        boolean[][] dp = new boolean[s.length()][s.length()]; // 表示以 j 结尾 i 开头的字符串是否为回文子串
        for (int i = 0; i < s.length(); i++) {
            dp[i][i] = true;
        }
        int res = 0;
        for (int i = s.length() - 1; i >= 0; i--) {  // 从下到上
//            int ans = 1;
            for (int j = i; j < s.length(); j++) {  // 从右到左
                if (s.charAt(i) == s.charAt(j)){
                    if (j - i < 3){  // 相等的情况下，长度为 0、1、2 都是回文串
                        dp[i][j] = true;
                    }else if (dp[i + 1][j - 1]){
                        dp[i][j] = true;
                    }
                }
                if (dp[i][j]){
//                    ans = Math.max(ans, j - i + 1);
                    res ++;
                }
            }
        }

        return res;
    }

    // 最长回文子序列
    public int longestPalindromeSubseq(String s) {
        if (s.length() == 1) return 1;

        int[][] dp = new int[s.length()][s.length()]; // 表示以 i 开头 j 结尾的字符串的回文子序列长度
        for (int i = 0; i < s.length(); i++) {
            dp[i][i] = 1;
        }
        for (int i = s.length() - 1; i >= 0; i--) {  // 从下到上
            for (int j = i+1; j < s.length(); j++) {  // 从右到左
                if (s.charAt(i) == s.charAt(j)){
                    dp[i][j] = dp[i + 1][j - 1] + 2;
                    }else{
                        dp[i][j] = Math.max(dp[i + 1][j], dp[i][j - 1]);
                    }
                }

            }

        return dp[0][s.length()-1];
    }



}


// n 叉树节点
class NodeN {
    public int val;
    public List<NodeN> children;

    public NodeN() {}

    public NodeN(int _val) {
        val = _val;
    }

    public NodeN(int _val, List<NodeN> _children) {
        val = _val;
        children = _children;
    }
};


// 设计单调队列
class MyPorQueue{
    Deque<Integer> deque;

    public MyPorQueue() {
        this.deque = new LinkedList<>();
    }

    public void poll(int val){
        if (!deque.isEmpty() && val == deque.peek()) {
            deque.poll();
        }
    }

    public void add(int value){
        while (!deque.isEmpty() && deque.getLast() < value){
            deque.removeLast();
        }
        deque.addLast(value);


    }

    public int peek(){
        return deque.peek();
    }
}



// 两个栈模拟队列
class MyQueue{
    private Stack<Integer> stack1;
    private Stack<Integer> stack2;
    public MyQueue() {
        stack1 = new Stack<>();
        stack2 = new Stack<>();
    }

    public void push(int x) {
        stack1.push(x);
    }

    public int pop() {
        if (!stack1.isEmpty()){
            while (!stack1.isEmpty()){
                stack2.push(stack1.pop());
            }
            int res = stack2.pop();
            while (!stack2.isEmpty()){
                stack1.push(stack2.pop());
            }
            return res;
        }else {
            return -1;
        }
    }

    public int peek() {
        if (!stack1.isEmpty()){
            while (!stack1.isEmpty()){
                stack2.push(stack1.pop());
            }
            int res = stack2.pop();
            stack2.push(res);
            while (!stack2.isEmpty()){
                stack1.push(stack2.pop());
            }
            return res;
        }else {
            return -1;
        }
    }

    public boolean empty() {
        return stack1.isEmpty();
    }

}


// 单链表节点
class ListNode_2 {
    int val;
    ListNode_2 next;
    ListNode_2() {}
    ListNode_2(int val) { this.val = val; }
    ListNode_2(int val, ListNode_2 next) { this.val = val; this.next = next; }
  }