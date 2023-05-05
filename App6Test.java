package com.xu;

import com.beust.ah.A;
import junit.framework.TestCase;

import java.util.ArrayList;
import java.util.Arrays;

public class App6Test extends TestCase {

    public void testSearch() {
        App6 app6 = new App6();
        System.out.println(app6.search(new int[]{-1, 0, 3, 5, 9, 12}, 9));
    }

    public void testSearch2() {
        App6 app6 = new App6();
        System.out.println(app6.search2(new int[]{-1, 0, 3, 5, 9, 12}, 9));
    }

    public void testSwapPairs() {
        App6 app6 = new App6();
        ListNode root = new ListNode(1);
        root.next = new ListNode(2);
        root.next.next = new ListNode(3);
        root.next.next.next = new ListNode(4);

        ListNode res = app6.swapPairs(root);
        while (res != null){
            System.out.println(res.val);
            res = res.next;
        }
    }

    public void testRemoveNthFromEnd() {
        App6 app6 = new App6();
        ListNode root = new ListNode(1);
        root.next = new ListNode(2);
        root.next.next = new ListNode(3);
        root.next.next.next = new ListNode(4);
        root.next.next.next.next = new ListNode(5);
        ListNode res = app6.removeNthFromEnd(root, 2);
        while (res != null){
            System.out.println(res.val);
            res = res.next;
        }
    }


    public void testIsAnagram2() {
        App6 app6 = new App6();
        System.out.println(app6.isAnagram2("a", "aa"));
    }

    public void testIsHappy() {
        App6 app6 = new App6();
        System.out.println(app6.isHappy(19));
    }

    public void testIsValid() {
        App6 app6 = new App6();
        System.out.println(app6.isValid("(){}}{"));
    }

    public void testRemoveDuplicates() {
        App6 app6 = new App6();
        System.out.println(app6.removeDuplicates("abbaca"));
    }

    public void testMaxSlidingWindow() {
        App6 app6 = new App6();
        System.out.println(Arrays.toString(app6.maxSlidingWindow(new int[]{7,2,4}, 2)));
    }

    public void testBuildTree() {
        App6 app6 = new App6();
        TreeNode root = app6.buildTree(new int[]{9,3,15,20,7}, new int[]{9,15,7,20,3});
        ArrayList<Integer> arrayList = new ArrayList<>();
        app6.preSearch(root, arrayList);
        for (Integer integer : arrayList) {
            System.out.println(integer);
        }
    }

    public void testPartition() {
        App6 app6 = new App6();
        System.out.println(app6.partition("aab"));
    }

    public void testRestoreIpAddresses() {
        App6 app6 = new App6();
//        System.out.println(app6.restoreIpAddresses("25525511135"));
        System.out.println(app6.restoreIpAddresses("101023"));
//        System.out.println(app6.restoreIpAddresses("99999999999999999999"));
    }

    public void testSubsets() {
        App6 app6 = new App6();
        System.out.println(app6.subsets(new int[]{1, 2, 3}));
    }

    public void testSubsetsWithDup() {
        App6 app6 = new App6();
        System.out.println(app6.subsetsWithDup(new int[]{1, 2, 2}));
//        System.out.println(app6.subsetsWithDup(new int[]{1, 1}));
    }

    public void testFindSubsequences() {
        App6 app6 = new App6();
//        System.out.println(app6.findSubsequences(new int[]{4, 6, 7, 7}));
        System.out.println(app6.findSubsequences(new int[]{1,2,3,4,5,6,7,8,9,10,1,1,1,1,1}));
    }

    public void testPermute() {
        App6 app6 = new App6();
        System.out.println(app6.permute(new int[]{1, 2, 3}));
    }

    public void testPermuteUnique() {
        App6 app6 = new App6();
        System.out.println(app6.permuteUnique(new int[]{1, 1, 2}));
    }

    public void testWiggleMaxLength() {
        App6 app6 = new App6();
//        System.out.println(app6.wiggleMaxLength(new int[]{1, 7, 4, 9, 2, 5}));
//        System.out.println(app6.wiggleMaxLength(new int[]{1,17,5,10,13,15,10,5,16,8}));
        System.out.println(app6.wiggleMaxLength(new int[]{1,2,3,4,5,6,7,8,9}));
//        System.out.println(app6.wiggleMaxLength(new int[]{0,0,0,0,0}));
    }

    public void testJump() {
        App6 app6 = new App6();
        System.out.println(app6.jump(new int[]{2, 3, 1, 1, 4}));
    }

    public void testMerge() {
        App6 app6 = new App6();
        System.out.println(Arrays.deepToString(app6.merge(new int[][]{{1, 3}, {2, 6}, {8, 10}, {15, 18}})));
    }

    public void testDailyTemperatures() {
        App6 app6 = new App6();
        System.out.println(Arrays.toString(app6.dailyTemperatures(new int[]{73, 74, 98, 75, 71, 69, 72, 76, 73})));
    }

    public void testNextGreaterElements() {
        App6 app6 = new App6();
        System.out.println(Arrays.toString(app6.nextGreaterElements(new int[]{1, 2, 3, 4, 3})));
    }
}