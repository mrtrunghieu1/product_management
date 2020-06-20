/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package productmanagement;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.swing.JTable;
import javax.swing.table.DefaultTableModel;
import productmanagement.productProperties;
/**
 *
 * @author Administrator
 */
public class MyDBLib {
    public Connection conn;
    public boolean dbConnect(String userName, String passwd)
    {
        boolean ret= false;
        try {
            conn = DriverManager.getConnection("jdbc:mysql://localhost:3306/db_product", userName, passwd);
            if(conn != null)
                ret=true;
        } catch (SQLException ex) {
            Logger.getLogger(MyDBLib.class.getName()).log(Level.SEVERE, null, ex);
        }
        return ret;
    }
    // Demo Test
//    public void listSys()
//    {
//        try {
//            Statement st = conn.createStatement();
//            ResultSet rs = st.executeQuery("select * from product.production");
//            System.out.println("ID, Name, CodeName, Number, Price");
//            while(rs.next())
//            {
//                System.out.println(rs.getString("id")+" "+rs.getString("name")+
//                rs.getString("code_prodution")+" "+rs.getString("numbers")+" "+rs.getString("price")
//                );
//            }
//            rs.close();
//            st.close();
//        } catch (SQLException ex) {
//            Logger.getLogger(MyDBLib.class.getName()).log(Level.SEVERE, null, ex);
//        }
//    }
    // Fetching database Prodution display on the Table
    public void fetch_production_db(String query, JTable table)
    {
        DefaultTableModel tableModel = (DefaultTableModel) table.getModel();
        tableModel.setRowCount(0);
        ArrayList<productProperties> list = new ArrayList<>();
        
        try {
            Statement st = conn.createStatement();
            ResultSet rs = st.executeQuery(query);
            while(rs.next())
            {
                productProperties product = new productProperties();
                product.setName(rs.getString("product_name"));
                product.setProduct_code(rs.getString("product_ID"));
                product.setProduct_number(rs.getInt("product_number"));
                product.setRetail_price(rs.getInt("product_retail_price"));
                product.setCategory(rs.getString("product_category"));
                list.add(product);
            }
        } catch (SQLException ex) {
            Logger.getLogger(MyDBLib.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        for (int i = 0; i < list.size(); i++)
        {
            String[] data = new String[5];
            data[0] = list.get(i).getName();
            data[1] = list.get(i).getProduct_code();
            data[2] = Integer.toString(list.get(i).getProduct_number());
            data[3] = Integer.toString(list.get(i).getRetail_price());
            data[4] = list.get(i).getCategory();
            tableModel.addRow(data);
        }
        table.setModel(tableModel);
        tableModel.fireTableDataChanged();
    }
    // Add product information to data base
    public void add_info_product_db(productProperties base_product) 
    {
        try {
//            System.out.println(base_product.getId(),base_product.getName(), base_product.getProduct_number());
            Statement st = conn.createStatement();
            String query = "INSERT INTO db_product.db_product(product_ID, product_name, "
                    + "product_number,product_retail_price, product_entry_price, product_category) "
                    + "VALUES (" + base_product.getId() +",'"+ base_product.getName()+"',"+ base_product.getProduct_number()+","
                    + base_product.getRetail_price()+","+ base_product.getEntry_price()+",'"+base_product.getCategory()+ "')"; 
            st.executeUpdate(query);
        } catch (SQLException ex) {
            Logger.getLogger(MyDBLib.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
   // Search product from database via text field search 
    public void searching_text_field_db(String text, JTable table)
    {
        DefaultTableModel tableModel = (DefaultTableModel) table.getModel();
        tableModel.setRowCount(0);
        ArrayList<productProperties> list = new ArrayList<>();
        try {
            Statement st = conn.createStatement();
            String query = "SELECT * FROM db_product.db_product WHERE product_name = '" + text + "'";
            ResultSet rs = st.executeQuery(query);
            
//            System.out.println(rs);
            while(rs.next())
            {
                productProperties product = new productProperties();
                product.setName(rs.getString("product_name"));
                product.setProduct_code(rs.getString("product_ID"));
                product.setProduct_number(rs.getInt("product_number"));
                product.setRetail_price(rs.getInt("product_retail_price"));
                product.setCategory(rs.getString("product_category"));
                list.add(product);
            }
        } catch (SQLException ex) {
            Logger.getLogger(MyDBLib.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        for (int i = 0; i < list.size(); i++)
        {
            String[] data = new String[5];
            data[0] = list.get(i).getName();
            data[1] = list.get(i).getProduct_code();
            data[2] = Integer.toString(list.get(i).getProduct_number());
            data[3] = Integer.toString(list.get(i).getRetail_price());
            data[4] = list.get(i).getCategory();
            tableModel.addRow(data);
        }
        table.setModel(tableModel);
        tableModel.fireTableDataChanged();
    }
}
