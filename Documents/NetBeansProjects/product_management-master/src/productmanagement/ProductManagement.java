/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package productmanagement;
/**
 *
 * @author Administrator
 */
public class ProductManagement {

    /**
     * @param args the command line arguments
     */
//    String queryData = "SELECT name, product_code, product_number, retail_price, category FROM product.db_product";
    public static void main(String[] args) {
//        MyDBLib db = new MyDBLib();
//        db.dbConnect("root", "h02111998");
//        db.fetch_production_db(queryData, jTable1);
        jPanelProduct jpp = new jPanelProduct();
        jpp.show();
    } 
}
